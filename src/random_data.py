import pandas as pd
from datetime import datetime, timedelta
import subprocess
import numpy as np
import sys

class RandomGenerator:
    def __init__(self, output_path):
        self.output_path = output_path
        self.data_file_path = f"{output_path}/sensor_data.csv"

        # Set seed for reproducibility
        np.random.seed(42)

        # Define date range for dummy data
        self.start_date = datetime(2023, 1, 1)
        self.end_date = datetime(2023, 1, 10)

    def random_dates(self, n=10):
        date_range = (self.end_date - self.start_date).days
        random_dates = [self.start_date + timedelta(days=np.random.randint(date_range)) for _ in range(n)]
        return sorted(random_dates)

    def generate_dummy_data(self, num_machines=5, num_sensors=3, freq='H'):
        machine_ids = [f'Machine_{i}' for i in range(1, num_machines + 1)]
        sensor_ids = [f'Sensor_{j}' for j in range(1, num_sensors + 1)]

        dates = pd.date_range(start=self.start_date, end=self.end_date, freq=freq)
        data = {'Timestamp': [], 'Machine_ID': [], 'Sensor_ID': [], 'Reading': []}

        for date in dates:
            for machine_id in machine_ids:
                for sensor_id in sensor_ids:
                    data['Timestamp'].append(date)
                    data['Machine_ID'].append(machine_id)
                    data['Sensor_ID'].append(sensor_id)
                    # Simulate sensor readings as random values
                    data['Reading'].append(np.random.normal(loc=100, scale=20))

        return pd.DataFrame(data)

    def generate_and_append_data(self, num_machines=5, num_sensors=3, freq='H'):
        try:
            existing_data = pd.read_csv(self.data_file_path)
        except FileNotFoundError:
            existing_data = pd.DataFrame(columns=['Timestamp', 'Machine_ID', 'Sensor_ID', 'Reading'])

        if not existing_data.empty and 'Timestamp' in existing_data.columns:
            existing_data['Timestamp'] = pd.to_datetime(existing_data['Timestamp'])
            start_date = existing_data['Timestamp'].max() + timedelta(hours=1)
        else:
            start_date = datetime.now()

        end_date = start_date + timedelta(days=1)

        new_data = self.generate_dummy_data(num_machines, num_sensors, freq)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)

        updated_data.to_csv(self.data_file_path, index=False)

if __name__ == "__main__":
    output_path = sys.argv[1]
    choice = sys.argv[2]

    data_generator = RandomGenerator(output_path)

    if choice == 'generate':
        print("Generating new data")
        data_generator.generate_dummy_data(num_machines=5, num_sensors=3)
    elif choice == 'append':
        print("Appending data")
        data_generator.generate_and_append_data(num_machines=5, num_sensors=3)
