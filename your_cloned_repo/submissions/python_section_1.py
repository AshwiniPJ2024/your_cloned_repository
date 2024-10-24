#question 1:
from typing import List, Dict

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    
    for i in range(0, len(lst), n):
        group_end = min(i + n, len(lst))
        group = lst[i:group_end]
        
        for j in range(len(group)):
            lst[i + j] = group[len(group) - 1 - j]
    
    return lst

#question 2:

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    
    length_dict = {}
    
    for string in lst:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []  
        length_dict[length].append(string)  
    
    return dict(sorted(length_dict.items()))
print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))  
# Output: {3: ['bat', 'car', 'dog'], 4: ['bear'], 5: ['apple'], 8: ['elephant']}

print(group_by_length(["one", "two", "three", "four"]))  
# Output: {3: ['one', 'two'], 4: ['four'], 5: ['three']}

#question 3:
from typing import Dict, Any

def flatten_dict(nested_dict: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    
    def _flatten(current, parent_key=''):
        items = []
        
        if isinstance(current, dict):
            for k, v in current.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                items.extend(_flatten(v, new_key).items())
        elif isinstance(current, list):
            for i, v in enumerate(current):
                new_key = f"{parent_key}[{i}]"
                items.extend(_flatten(v, new_key).items())
        else:
            items.append((parent_key, current))
        return dict(items)

    
    return _flatten(nested_dict)

# Usage Example
nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

flattened_dict = flatten_dict(nested_dict)
print(flattened_dict)

#question 4:

def unique_permutations(nums: List[int]) -> List[List[int]]:
    
    def backtrack(path, used):
        
        if len(path) == len(nums):
            result.append(path[:])
            return
        
        for i in range(len(nums)):
            
            if used[i] or (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]):
                continue

            used[i] = True
            path.append(nums[i])
            backtrack(path, used)

            path.pop()
            used[i] = False

    nums.sort()
    result = []
    used = [False] * len(nums)

    
    backtrack([], used)
    return result

#Example
input_list = [1, 1, 2]
output = unique_permutations(input_list)
print(output)

#question 5:
import re

def find_all_dates(text: str) -> List[str]:
    
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  
        r'\b\d{2}/\d{2}/\d{4}\b',  
        r'\b\d{4}\.\d{2}\.\d{2}\b'  
    ]

    combined_pattern = '|'.join(date_patterns)

    matches = re.findall(combined_pattern, text)

    return matches

#Example
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
output = find_all_dates(text)
print(output)

#question 6:
import polyline
import pandas as pd
import math

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    
    R = 6371000  # Earth's radius in meters

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    #Haversine formula
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    
    coordinates = polyline.decode(polyline_str)

    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    distances = [0]  

    for i in range(1, len(df)):
        lat1, lon1 = df.iloc[i - 1][['latitude', 'longitude']]
        lat2, lon2 = df.iloc[i][['latitude', 'longitude']]
        distance = haversine(lat1, lon1, lat2, lon2)
        distances.append(distance)

    df['distance'] = distances

    return df

#example
polyline_str = '_p~iF~ps|U_ulLnnqC_mqNvxq`@'
df = polyline_to_dataframe(polyline_str)
print(df)

#question 7:

def rotate_and_transform_matrix(matrix: List[List[int]]) -> List[List[int]]:
    
    n = len(matrix)  

    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]

    transformed_matrix = [[0] * n for _ in range(n)]  

    for i in range(n):
        for j in range(n):
    
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            transformed_matrix[i][j] = row_sum + col_sum

    return transformed_matrix

#example
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
result = rotate_and_transform_matrix(matrix)
for row in result:
    print(row)

#question 8:
    import pandas as pd
from datetime import datetime, time

def time_check(df: pd.DataFrame) -> pd.Series:
    
    valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday", 
                  "Friday", "Saturday", "Sunday"]

    completeness = []

    grouped = df.groupby(['id', 'id_2'])

    for (id_, id_2), group in grouped:
        
        covered_days = set()
        complete_24_hour_coverage = True

        for _, row in group.iterrows():
    
            start_day = row['startDay']
            end_day = row['endDay']

            if start_day not in valid_days or end_day not in valid_days:
                raise ValueError(f"Invalid day: {start_day} or {end_day}")

            start_idx = valid_days.index(start_day)
            end_idx = valid_days.index(end_day) + 1
            covered_days.update(valid_days[start_idx:end_idx])

            start_time = datetime.strptime(row['startTime'], "%H:%M:%S").time()
            end_time = datetime.strptime(row['endTime'], "%H:%M:%S").time()
            if not (start_time == time(0, 0, 0) and end_time == time(23, 59, 59)):
                complete_24_hour_coverage = False

        all_days_covered = len(covered_days) == 7

        is_incomplete = not (all_days_covered and complete_24_hour_coverage)
        completeness.append(((id_, id_2), is_incomplete))

    
    index = pd.MultiIndex.from_tuples([x[0] for x in completeness], names=["id", "id_2"])
    result_series = pd.Series([x[1] for x in completeness], index=index)

    return result_series

#example
df = pd.read_csv('dataset-1.csv')  
result = time_check(df)
print(result)

