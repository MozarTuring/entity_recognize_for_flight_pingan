import json

def generate_pattern(file_path, key, pattern):
    """

    :param file_path:
    :param key:
    :param pattern: [(pattern_prefix, pattern_suffix), ...]
    :return:
    """
    with open(file_path, mode='r', encoding='utf-8') as f:
        flight_info = json.load(f)
    total_flight_code = flight_info[key]
    flight_pattern = []
    for i in total_flight_code:
        for pattern_prefix, pattern_suffix in pattern:
            pattern_tmp = pattern_prefix + i + pattern_suffix
            flight_pattern.append(pattern_tmp)
    return flight_pattern