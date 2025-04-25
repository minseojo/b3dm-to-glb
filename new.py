import struct
import json
import os
import re
import math
import numpy as np

# --- 위경도 → ECEF 좌표 변환 ---
def lonlat_to_ecef_geodetic(lon_deg, lat_deg, height=0.0):
    # WGS84 타원체 파라미터
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 2 * f - f * f
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    N = a / math.sqrt(1 - e2 * sin_lat**2)
    # ECEF 좌표
    x_s = N * cos_lat * cos_lon
    y_s = N * cos_lat * sin_lon
    z_s = N * (1 - e2) * sin_lat
    return [x_s, y_s, z_s]

# --- ENU → GLB 좌표계 회전 쿼터니언 ---
def enu_to_glb_rotation_quaternion():
    return [-0.7071067811865475, 0.0, 0.0, 0.7071067811865475]

# --- 패딩 함수 (4바이트 단위) ---
def pad_to_4bytes_space(data: bytes) -> bytes:
    padding = (4 - (len(data) % 4)) % 4
    return data + b'\x20' * padding

def pad_to_4bytes_zero(data: bytes) -> bytes:
    padding = (4 - (len(data) % 4)) % 4
    return data + b'\x00' * padding

# --- B3DM 파일 파싱 (FeatureTable, BatchTable, GLB 추출) ---
def extract_b3dm_components(b3dm_path):
    with open(b3dm_path, "rb") as f:
        header = f.read(28)
        _, _, _, ft_json_len, ft_bin_len, bt_json_len, bt_bin_len = struct.unpack('<4sIIIIII', header)
        ft_json = f.read(ft_json_len)
        f.read(ft_bin_len)
        bt_json = f.read(bt_json_len)
        f.read(bt_bin_len)
        glb_data = f.read()
        return {
            "feature_table_json": json.loads(ft_json.decode("utf-8")),
            "batch_table_json": json.loads(bt_json.decode("utf-8")),
            "glb_data": glb_data
        }

# --- GLB 내부 Chunk 분리 ---
def extract_glb_chunks(glb_data):
    header = glb_data[:12]
    magic, version, length = struct.unpack("<III", header)
    assert magic == 0x46546C67

    json_len, json_type = struct.unpack("<I4s", glb_data[12:20])
    json_chunk = glb_data[20:20 + json_len]

    bin_header_start = 20 + json_len
    bin_len, bin_type = struct.unpack("<I4s", glb_data[bin_header_start:bin_header_start + 8])
    bin_start = bin_header_start + 8  # ✅ 실제 binary data 시작점
    bin_chunk = glb_data[bin_start : bin_start + bin_len]

    print(f"[DEBUG] bin_len: {bin_len}, bin_type: {bin_type}")
    print(f"[DEBUG] actual bin_chunk size: {len(bin_chunk)}")
    if len(bin_chunk) < bin_len:
        print(f"⚠️ bin_chunk too short: expected {bin_len}, got {len(bin_chunk)}")

    return json.loads(json_chunk.decode("utf-8")), bin_chunk

def add_mesh_features_from_existing_batchid(glb_json, glb_bin):
    for mesh in glb_json.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            attributes = primitive.setdefault("attributes", {})
            extensions = primitive.setdefault("extensions", {})

            if "_BATCHID" not in attributes:
                raise ValueError("primitive에 _BATCHID가 없습니다.")
            batch_accessor_index = attributes.pop("_BATCHID")
            attributes["_FEATURE_ID_0"] = batch_accessor_index

            draco = extensions.get("KHR_draco_mesh_compression", {}).get("attributes", {})
            if "_BATCHID" in draco:
                draco["_FEATURE_ID_0"] = draco.pop("_BATCHID")

            accessor = glb_json["accessors"][batch_accessor_index]
            max_id = accessor.get("max", [0])[0]
            feature_count = int(max_id) + 1

            feature_id_attr_index = None
            for k in attributes.keys():
                match = re.match(r"_FEATURE_ID_(\d+)", k)
                if match and attributes[k] == batch_accessor_index:
                    feature_id_attr_index = int(match.group(1))
                    break

            if feature_id_attr_index is None:
                raise ValueError(
                    f"❌ primitive.attributes에서 batch_accessor_index={batch_accessor_index}에 해당하는 _FEATURE_ID_* key를 찾을 수 없습니다.")

            # ✅ EXT_mesh_features 등록 시 올바른 attribute index 사용
            extensions["EXT_mesh_features"] = {
                "featureIds": [{
                    "attribute": feature_id_attr_index,
                    "featureCount": feature_count,
                    "propertyTable": 0
                }]
            }

    return bytes(glb_bin)

def update_feature_id_attribute_indices(glb_json):
    for mesh in glb_json.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            attributes = primitive.get("attributes", {})
            ext = primitive.get("extensions", {}).get("EXT_mesh_features", {})
            feature_ids = ext.get("featureIds", [])

            for fid in feature_ids:
                target_index = None
                for key, val in attributes.items():
                    if key.startswith("_FEATURE_ID_"):
                        try:
                            if "attribute" not in fid or fid["attribute"] != val:
                                suffix = int(key.split("_FEATURE_ID_")[1])
                                if val == attributes[key]:
                                    target_index = suffix
                                    break
                        except ValueError:
                            continue

                if target_index is not None:
                    fid["attribute"] = target_index
                else:
                    raise ValueError(f"Could not find _FEATURE_ID_* attribute index for accessor {fid.get('attribute')}")

    return glb_json

def apply_avg_translation_and_rotation_to_nodes(glb_json, batch_table_json):
    lons = batch_table_json.get("MODEL_LON")
    lats = batch_table_json.get("MODEL_LAT")
    if not lons or not lats:
        print("⚠️ MODEL_LON / MODEL_LAT 누락되어 위치 보정 생략")
        return glb_json

    translations = []
    for lon, lat in zip(lons, lats):
        ground_ecef = lonlat_to_ecef_geodetic(lon, lat, 0.0)
        model_ecef = [ground_ecef[0], ground_ecef[2], -ground_ecef[1]]
        translations.append(model_ecef)

    avg = [sum(t[i] for t in translations) / len(translations) for i in range(3)]

    # 실제 루트 노드만 찾아서 children으로 설정
    all_children = set()
    for node in glb_json.get("nodes", []):
        for child in node.get("children", []):
            all_children.add(child)

    root_nodes = [i for i in range(len(glb_json.get("nodes", []))) if i not in all_children]

    new_node = {
        "translation": avg,
        "rotation": enu_to_glb_rotation_quaternion(),
        "scale": [1, 1, 1],
        "children": root_nodes
    }

    glb_json.setdefault("nodes", []).append(new_node)
    glb_json.setdefault("scenes", [{}])[glb_json.get("scene", 0)]["nodes"] = [len(glb_json["nodes"]) - 1]

    return glb_json

def generate_metadata_buffers_dynamic_types(batch_table, bin_chunk, glb_json):
    new_bin = bytearray(bin_chunk)
    buffer_views = glb_json.setdefault("bufferViews", [])
    offset = len(new_bin)

    def append_buffer(data: bytes, align: int):
        nonlocal offset
        pad_before = (align - (offset % align)) % align
        offset += pad_before
        new_bin.extend(b'\x00' * pad_before)

        start = offset
        new_bin.extend(data)
        offset += len(data)

        # 끝에도 align 맞춰줘야 다음 bufferView가 맞게 시작함
        pad_after = (4 - (offset % 4)) % 4
        if pad_after:
            new_bin.extend(b'\x00' * pad_after)
            offset += pad_after

        return start, len(data)

    # === EXT_structural_metadata 설정 ===
    ext = glb_json["extensions"]["EXT_structural_metadata"]
    table = ext["propertyTables"][0]["properties"]
    table.clear()

    # === schema 업데이트 ===
    schema = ext["schema"]["classes"]["class_batch_table"]["properties"]
    schema.clear()

    for key, values in batch_table.items():

        if all(isinstance(v, str) for v in values):
            string_offsets = []
            current = 0
            encoded_chunks = []

            for s in values:
                encoded = s.encode("utf-8")
                string_offsets.append(current)
                encoded_chunks.append(encoded)
                current += len(encoded)

            # 마지막 오프셋 추가
            string_offsets.append(current)
            string_pool = b''.join(encoded_chunks)

            assert len(string_offsets) == len(values) + 1, \
                f"⚠️ stringOffsets 길이({len(string_offsets)}) ≠ 문자열 개수({len(values)})"

            # 1바이트 정렬로 pool 저장
            pool_start, pool_len = append_buffer(string_pool, align=1)

            # 4바이트 정렬로 offset 저장
            ooffset_array = np.array(string_offsets, dtype=np.uint32).tobytes()

            offsets_start, offsets_len = append_buffer(ooffset_array, align=4)

            bv_index_value = len(buffer_views)
            buffer_views.append({"buffer": 0, "byteOffset": pool_start, "byteLength": pool_len})

            bv_index_offsets = len(buffer_views)
            buffer_views.append({"buffer": 0, "byteOffset": offsets_start, "byteLength": offsets_len})

            table[key] = {
                "values": bv_index_value,
                "stringOffsets": bv_index_offsets
            }

            schema[key] = {
                "name": key,
                "description": f"Generated from {key}",
                "type": "STRING",
                "required": True
            }
        else:
            comp_type, np_type = get_component_type_and_dtype(values)
            data_bytes = np.array(values, dtype=np_type).tobytes()
            start, length = append_buffer(data_bytes, align=4)

            bv_index = len(buffer_views)
            buffer_views.append({"buffer": 0, "byteOffset": start, "byteLength": length})

            table[key] = {"values": bv_index}

            schema[key] = {
                "name": key,
                "description": f"Generated from {key}",
                "type": "SCALAR",
                "componentType": {
                    5121: "UINT8", 5123: "UINT16", 5125: "UINT32", 5126: "FLOAT32"
                }[comp_type],
                "required": True
            }

    return bytes(new_bin)

# --- b3d metadata 삽입 ---
def update_glb_extensions(glb_json, batch_table):
    for ext in ["EXT_structural_metadata", "EXT_mesh_features", "KHR_draco_mesh_compression", "KHR_materials_unlit"]:
        if ext not in glb_json.get("extensionsUsed", []):
            glb_json.setdefault("extensionsUsed", []).append(ext)
    if "KHR_draco_mesh_compression" not in glb_json.get("extensionsRequired", []):
        glb_json.setdefault("extensionsRequired", []).append("KHR_draco_mesh_compression")

    count = len(next(iter(batch_table.values())))  # 어떤 필드든 길이는 동일하므로 하나만 참조

    schema_properties = {}
    for key, values in batch_table.items():

        if all(isinstance(v, str) for v in values):
            schema_properties[key] = {
                "name": key,
                "description": f"Generated from {key}",
                "type": "STRING",
                "required": True
            }
        else:
            _, np_type = get_component_type_and_dtype(values)
            if np_type == np.uint8:
                ctype = "UINT8"
            elif np_type == np.uint16:
                ctype = "UINT16"
            elif np_type == np.uint32:
                ctype = "UINT32"
            elif np_type == np.float32:
                ctype = "FLOAT32"
            else:
                raise ValueError(f"❌ 알 수 없는 타입: {key} → {np_type}")

            schema_properties[key] = {
                "name": key,
                "description": f"Generated from {key}",
                "type": "SCALAR",
                "componentType": ctype,
                "required": True
            }

    structural_metadata = {
        "schema": {
            "id": "ID_batch_table",
            "name": "Generated from batch_table",
            "classes": {
                "class_batch_table": {
                    "name": "Generated from batch_table",
                    "properties": schema_properties
                }
            }
        },
        "propertyTables": [
            {
                "class": "class_batch_table",
                "count": count,
                "properties": {}  # 실제 binary offset은 generate_metadata_buffers_dynamic_types에서 설정
            }
        ]
    }

    glb_json.setdefault("extensions", {})["EXT_structural_metadata"] = structural_metadata
    return glb_json

def get_component_type_and_dtype(values):
    if all(isinstance(v, str) for v in values):
        return "STRING", None
    elif any(isinstance(v, float) or isinstance(v, np.floating) for v in values):
        return 5126, np.float32

    max_val = max(values)
    if max_val <= 255:
        return 5121, np.uint8
    elif max_val <= 65535:
        return 5123, np.uint16
    else:
        return 5125, np.uint32

def assign_feature_count_using_accessor_max(glb_json):
    for mesh in glb_json.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            attrs = primitive.get("attributes", {})
            batch_accessor_index = attrs.get("_FEATURE_ID_0")
            if batch_accessor_index is None:
                continue

            accessor = glb_json["accessors"][batch_accessor_index]

            if "max" in accessor:
                feature_count = int(accessor["max"][0]) - int(accessor["min"][0]) + 1
            else:
                print(f"⚠️ accessor[{batch_accessor_index}]에 max 값이 없어 featureCount 계산 생략")
                continue

            # 설정
            feature_ids = primitive.get("extensions", {}).get("EXT_mesh_features", {}).get("featureIds", [])
            for f in feature_ids:
                f["featureCount"] = feature_count
    return glb_json

def remove_all_node_matrices(gltf_json):
    for node in gltf_json.get("nodes", []):
        node.pop("matrix", None)

def clean_min_max_except_position(glb_json):
    position_accessor_indices = set()

    # 1. POSITION accessor 인덱스 수집
    for mesh in glb_json.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            attrs = primitive.get("attributes", {})
            pos_idx = attrs.get("POSITION")
            if pos_idx is not None:
                position_accessor_indices.add(pos_idx)

    for i, accessor in enumerate(glb_json.get("accessors", [])):
        # ✅ POSITION accessor면 유지
        if i in position_accessor_indices:
            continue
        # ✅ 나머지는 제거
        accessor.pop("min", None)
        accessor.pop("max", None)

def write_glb(glb_json, final_bin: bytes, output_path):
    bin_padded = pad_to_4bytes_zero(final_bin)
    glb_json["buffers"] = [{"byteLength": len(bin_padded)}]

    remove_all_node_matrices(glb_json)
    json_bytes = json.dumps(glb_json, separators=(",", ":")).encode("utf-8")
    json_padded = pad_to_4bytes_space(json_bytes)

    total_len = 12 + 8 + len(json_padded) + 8 + len(bin_padded)

    with open(output_path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total_len))
        f.write(struct.pack("<I4s", len(json_padded), b'JSON'))
        f.write(json_padded)
        f.write(struct.pack("<I4s", len(bin_padded), b'BIN\0'))
        f.write(bin_padded)

# === 실행 순서 ===
def convert_b3dm_to_glb_with_metadata(b3dm_path, output_path):
    # B3DM 구성 요소 추출
    parts = extract_b3dm_components(b3dm_path)
    glb_json, bin_chunk = extract_glb_chunks(parts["glb_data"])

    # 모델 위치, 회전 적용
    apply_avg_translation_and_rotation_to_nodes(glb_json, parts["batch_table_json"])

    # GLB 확장 초기화 및 schema 설정
    updated_json = update_glb_extensions(glb_json, parts["batch_table_json"])

    # 메타데이터 데이터 + binary buffer 추가
    final_bin = generate_metadata_buffers_dynamic_types(parts["batch_table_json"], bin_chunk, updated_json)

    # 기존 _BATCHID를 _FEATURE_ID_0으로 전환
    final_bin = add_mesh_features_from_existing_batchid(updated_json, final_bin)

    # featureCount 정확히 설정
    updated_json = assign_feature_count_using_accessor_max(updated_json)
    updated_json = update_feature_id_attribute_indices(updated_json)

    # 불필요한 matrix 제거 및 min/max 정리
    remove_all_node_matrices(updated_json)
    clean_min_max_except_position(updated_json)

    # 최종 GLB 작성
    write_glb(updated_json, final_bin, output_path)

    print(f"✅ metadata 포함 GLB 저장 완료: {output_path}")

# === 디렉토리 내 전체 변환 ===
def convert_all_b3dm_in_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".b3dm"):
            input_path = os.path.join(input_dir, filename)
            match = re.match(r"B_(\d+)_(\d+)_(\d+)\.b3dm", filename)
            if not match:
                print(f"❌ 무시됨 (이름 형식 아님): {filename}")
                continue
            level, x, y = match.groups()
            output_filename = f"content_{level}_{x}_{y}.glb"
            output_path = os.path.join(output_dir, output_filename)
            try:
                convert_b3dm_to_glb_with_metadata(input_path, output_path)
            except Exception as e:
                print(f"❌ 변환 실패: {filename} → {e}")

# === 변환 시작 ===
convert_all_b3dm_in_folder(
    input_dir=r"D:\implicit_tiling\example\server\tilsetjsonServer\src\main\resources\static\earth\songpa\sample\18_6_001\b3dm",
    output_dir=r"D:\implicit_tiling\example\server\tilsetjsonServer\src\main\resources\static\earth\songpa\sample\18_6_001\content"
)