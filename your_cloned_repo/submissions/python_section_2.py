#question 9:
import pandas as pd
import numpy as np

def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    
    start_col, end_col, distance_col = df.columns[:3]

    locations = pd.unique(df[[start_col, end_col]].values.ravel('K'))

    
    dist_matrix = pd.DataFrame(
        np.inf, index=locations, columns=locations
    )
    np.fill_diagonal(dist_matrix.values, 0)

    
    for _, row in df.iterrows():
        start, end, distance = row[start_col], row[end_col], row[distance_col]
        dist_matrix.loc[start, end] = distance
        dist_matrix.loc[end, start] = distance

    
    for k in locations:
        for i in locations:
            for j in locations:
                dist_matrix.loc[i, j] = min(
                    dist_matrix.loc[i, j],
                    dist_matrix.loc[i, k] + dist_matrix.loc[k, j]
                )

    return dist_matrix

# Usage Example
df = pd.read_csv('dataset-2.csv')
distance_matrix = calculate_distance_matrix(df)
print(distance_matrix)

#question 10:

def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    
    records = []

    
    for id_start in df.index:
        for id_end in df.columns:
            if id_start != id_end:  
                distance = df.loc[id_start, id_end]
                records.append((id_start, id_end, distance))

    unrolled_df = pd.DataFrame(records, columns=['id_start', 'id_end', 'distance'])

    unrolled_df = unrolled_df.sort_values(['id_start', 'id_end']).reset_index(drop=True)

    return unrolled_df


#question 11:

def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    
    avg_distances = df.groupby('id_start')['distance'].mean()

    
    reference_avg = avg_distances[reference_id]


    lower_bound = reference_avg * 0.90
    upper_bound = reference_avg * 1.10

    ids_within_threshold = avg_distances[
        (avg_distances >= lower_bound) & (avg_distances <= upper_bound)
    ].index

    result_df = pd.DataFrame(ids_within_threshold, columns=['id_start']).sort_values('id_start').reset_index(drop=True)

    return result_df

# Example
data = {
    'id_start': [1001400, 1001400, 1001400, 1001402, 1001402, 1001404],
    'id_end': [1001402, 1001404, 1001406, 1001404, 1001406, 1001406],
    'distance': [9.7, 29.9, 45.9, 20.2, 36.2, 16.0]
}

df = pd.DataFrame(data)

reference_id = 1001400
result_df = find_ids_within_ten_percentage_threshold(df, reference_id)

print(result_df)

#question 12:

def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    for vehicle, rate in rates.items():
        df[vehicle] = df['distance'] * rate

    df = df.drop(columns=['distance'])

    return df

# Example 
data = {
    'id_start': [1001400, 1001400, 1001402],
    'id_end': [1001402, 1001404, 1001404],
    'distance': [9.7, 29.9, 20.2]
}

df = pd.DataFrame(data)

result_df = calculate_toll_rate(df)

print(result_df)



#question 13:
from datetime import time

def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    weekday_intervals = {
        ('00:00:00', '10:00:00'): 0.8, 
        ('10:00:00', '18:00:00'): 1.2, 
        ('18:00:00', '23:59:59'): 0.8
    }

    weekend_discount = 0.7

    results = []

    for _, row in df.iterrows():
        
        for day in days_of_week[:5]:  
            for (start, end), discount in weekday_intervals.items():
                results.append(generate_toll_row(row, day, start, end, discount))

        
        for day in days_of_week[5:]:  
            results.append(generate_toll_row(row, day, '00:00:00', '23:59:59', weekend_discount))

    
    final_df = pd.DataFrame(results)
    return final_df

def generate_toll_row(row, day, start_time, end_time, discount):
    
    return {
        'id_start': row['id_start'],
        'id_end': row['id_end'],
        'distance': row['distance'],
        'start_day': day,
        'start_time': time.fromisoformat(start_time),
        'end_day': day,
        'end_time': time.fromisoformat(end_time),
        'moto': round(row['moto'] * discount, 2),
        'car': round(row['car'] * discount, 2),
        'rv': round(row['rv'] * discount, 2),
        'bus': round(row['bus'] * discount, 2),
        'truck': round(row['truck'] * discount, 2)
    }

# Example 
data = {
    'id_start': [1001400, 1001408],
    'id_end': [1001402, 1001410],
    'distance': [9.7, 11.1],
    'moto': [7.76, 8.88],
    'car': [11.64, 13.32],
    'rv': [14.55, 16.65],
    'bus': [21.34, 24.42],
    'truck': [34.92, 39.96]
}

df = pd.DataFrame(data)
result_df = calculate_time_based_toll_rates(df)
print(result_df)




