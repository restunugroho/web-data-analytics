import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class DataGenerators:
    @staticmethod
    def generate_manufacturing_data():
        logging.info("✅ generate manufacturing data")
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', '2024-12-31', freq='H')
        
        base_production = 850
        data = []
        
        for i, date in enumerate(dates):
            hour_multiplier = 1.3 if 8 <= date.hour <= 18 else 0.7
            day_multiplier = 0.4 if date.weekday() >= 5 else 1.0
            seasonal_multiplier = 1.2 if date.month in [6, 7, 8] else 1.0
            maintenance = 0.1 if np.random.random() < 0.05 else 1.0
            
            production = base_production * hour_multiplier * day_multiplier * seasonal_multiplier * maintenance
            production += np.random.normal(0, 50)
            production = max(0, production)
            
            defect_rate = max(0, min(15, np.random.normal(2.5, 1.5)))
            temperature = np.random.normal(75, 5)
            
            data.append({
                'timestamp': date.strftime('%d/%m/%Y %H:%M'),
                'production_units': round(production, 1),
                'defect_rate_percent': round(defect_rate, 2),
                'machine_temp_celsius': round(temperature, 1),
                'shift': 'Day' if 6 <= date.hour < 14 else 'Evening' if 14 <= date.hour < 22 else 'Night',
                'machine_id': f'M{(i % 5) + 1:03d}'
            })
        
        return pd.DataFrame(data)

    @staticmethod
    def generate_trading_data():
        logging.info("✅ generate trading data")
        np.random.seed(123)
        
        start_date = datetime(2024, 1, 1, 9, 30)
        data = []
        base_price = 100
        current_price = base_price
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        for days in range(180):
            trading_date = start_date + timedelta(days=days)
            
            if trading_date.weekday() >= 5:
                continue
                
            for minutes in range(0, 390, 5):
                timestamp = trading_date + timedelta(minutes=minutes)
                
                for symbol in symbols:
                    price_change = np.random.normal(0, 0.02) * current_price
                    current_price = max(10, current_price + price_change)
                    
                    hour = timestamp.hour
                    base_volume = np.random.exponential(50000 if hour in [9, 15] else 20000)
                    volume = int(base_volume)
                    
                    data.append({
                        'trade_date': timestamp.strftime('%Y%m%d'),
                        'trade_time': timestamp.strftime('%H:%M:%S'),
                        'symbol': symbol,
                        'price': round(current_price, 2),
                        'volume': volume,
                        'bid_ask_spread': round(np.random.uniform(0.01, 0.05), 3),
                        'market_cap_category': np.random.choice(['Large', 'Mid', 'Small'], p=[0.6, 0.3, 0.1])
                    })
                    
                    current_price = base_price + np.random.normal(0, 20)
        
        return pd.DataFrame(data)

    @staticmethod
    def generate_healthcare_data():
        logging.info("✅ generate healthcare data")
        np.random.seed(456)
        
        dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
        data = []
        departments = ['Emergency', 'Cardiology', 'Orthopedics', 'Pediatrics', 'ICU']
        age_groups = ['0-18', '19-35', '36-55', '56-75', '75+']
        
        for date in dates:
            base_admissions = 45 if date.weekday() < 5 else 65
            seasonal_factor = 1.4 if date.month in [12, 1, 2, 3] else 1.0
            daily_admissions = int(base_admissions * seasonal_factor * np.random.uniform(0.7, 1.3))
            
            for _ in range(daily_admissions):
                admission_hour = max(0, min(23, int(np.random.exponential(12))))
                department = np.random.choice(departments, p=[0.35, 0.15, 0.15, 0.2, 0.15])
                age_group = np.random.choice(age_groups, p=[0.15, 0.25, 0.25, 0.25, 0.1])
                
                los_map = {'Emergency': 2, 'ICU': 7, 'Cardiology': 4, 'Orthopedics': 4, 'Pediatrics': 4}
                los = max(1, int(np.random.exponential(los_map.get(department, 4))))
                
                base_cost_map = {'Emergency': 2500, 'Cardiology': 8500, 'Orthopedics': 12000, 
                               'Pediatrics': 4500, 'ICU': 15000}
                base_cost = base_cost_map[department]
                total_cost = base_cost + (los * 1200) + np.random.normal(0, 1000)
                total_cost = max(1000, total_cost)
                
                admission_datetime = date + timedelta(hours=admission_hour, minutes=np.random.randint(0, 60))
                
                data.append({
                    'admission_date': admission_datetime.strftime('%m-%d-%Y'),
                    'admission_time': admission_datetime.strftime('%H:%M'),
                    'department': department,
                    'patient_age_group': age_group,
                    'length_of_stay_days': los,
                    'total_cost_usd': round(total_cost, 2),
                    'discharge_status': np.random.choice(['Home', 'Transfer', 'AMA'], p=[0.85, 0.12, 0.03]),
                    'insurance_type': np.random.choice(['Private', 'Medicare', 'Medicaid', 'Uninsured'], 
                                                     p=[0.45, 0.25, 0.2, 0.1])
                })
        
        return pd.DataFrame(data)

    @staticmethod
    def generate_flight_data():
        logging.info("✅ generate flight data")
        np.random.seed(789)
        
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2024, 12, 31)
        dates = pd.date_range(start_date, end_date, freq='D')
        
        data = []
        base_passengers = 15000
        airlines = ['Garuda Indonesia']
        aircraft_types = ['Boeing 737']
        routes = ['CGK-DPS', 'CGK-SBY']

        for i, date in enumerate(dates):
            trend_multiplier = 1 + (i / len(dates)) * 0.8
            
            if date.weekday() in [4, 5, 6]:
                weekly_multiplier = 1.4
            else:
                weekly_multiplier = 1.0
            
            month = date.month
            seasonal_map = {6: 1.6, 7: 1.6, 12: 1.6, 1: 1.3, 2: 1.3, 4: 1.2, 5: 1.2, 
                          3: 1.1, 9: 1.1, 10: 1.1}
            seasonal_multiplier = seasonal_map.get(month, 0.8)
            
            holiday_boost = 1.0
            if (date.month == 12 and date.day >= 20) or (date.month == 1 and date.day <= 5):
                holiday_boost = 1.8
            elif date.month == 7 and 15 <= date.day <= 31:
                holiday_boost = 1.5
            
            daily_passengers = base_passengers * trend_multiplier * weekly_multiplier * seasonal_multiplier * holiday_boost
            daily_passengers += np.random.normal(0, daily_passengers * 0.1)
            daily_passengers = max(5000, int(daily_passengers))
            
            estimated_flights = max(30, int(daily_passengers / 180))
            
            for flight_num in range(estimated_flights):
                departure_hour = np.random.choice(range(5, 23), p=[0.02, 0.03, 0.08, 0.12, 0.15, 0.12, 
                                                       0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 
                                                       0.02, 0.02, 0.02, 0.02, 0.02, 0.02])
                departure_minute = np.random.randint(0, 60)
                
                airline = np.random.choice(airlines)
                aircraft = np.random.choice(aircraft_types)
                route = np.random.choice(routes)
                
                max_capacity = 189  # Boeing 737
                
                base_load_factor = 0.75
                if date.weekday() >= 5:
                    load_factor = min(0.95, base_load_factor + 0.15)
                else:
                    load_factor = base_load_factor
                
                load_factor *= seasonal_multiplier / 1.2
                load_factor = min(0.98, max(0.4, load_factor))
                
                passengers = int(max_capacity * load_factor)
                revenue_per_passenger = max(300000, np.random.normal(750000, 150000))
                fuel_cost = max(15000000, np.random.normal(25000000, 5000000))
                
                departure_time = datetime.combine(date.date(), datetime.min.time()).replace(
                    hour=departure_hour, minute=departure_minute)
                
                data.append({
                    'flight_date': date.strftime('%Y-%m-%d'),
                    'departure_time': departure_time.strftime('%H:%M'),
                    'airline': airline,
                    'aircraft_type': aircraft,
                    'route': route,
                    'passengers_count': passengers,
                    'max_capacity': max_capacity,
                    'load_factor_percent': round(load_factor * 100, 1),
                    'revenue_idr': int(passengers * revenue_per_passenger),
                    'fuel_cost_idr': int(fuel_cost),
                    'profit_idr': int(passengers * revenue_per_passenger - fuel_cost),
                    'flight_duration_hours': round(np.random.uniform(1.0, 2.5), 1),
                    'delay_minutes': max(0, int(np.random.exponential(15)))
                })
        
        return pd.DataFrame(data)

    @staticmethod
    def generate_school_book_data():
        logging.info("✅ generate school book data")
        np.random.seed(321)
        
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2024, 12, 31)
        dates = pd.date_range(start_date, end_date, freq='D')
        
        data = []
        base_sales = 500
        book_categories = ['Elementary Textbooks', 'Middle School Books', 'High School Books', 
                          'University Books', 'Activity Books', 'Reference Books']
        subjects = ['Mathematics', 'Science', 'Indonesian Language', 'English', 'Social Studies', 
                   'Religion', 'Arts', 'Physical Education']
        publishers = ['Erlangga', 'Tiga Serangkai', 'Yudhistira', 'Esis', 'Grafindo', 'Bumi Aksara']
        
        for date in enumerate(dates):
            date = date[1]
            month = date.month
            
            seasonal_map = {6: 8.0, 7: 12.0, 12: 4.0, 1: 5.0, 5: 2.0, 11: 2.0, 
                          2: 1.5, 8: 1.5, 3: 0.8, 4: 0.8, 9: 0.8, 10: 0.8}
            seasonal_multiplier = seasonal_map.get(month, 0.3)
            
            if date.weekday() >= 5:
                weekly_multiplier = 0.8 if seasonal_multiplier > 3 else 0.3
            else:
                weekly_multiplier = 1.0
            
            year_multiplier = 1 + (date.year - 2022) * 0.12
            
            daily_sales = base_sales * seasonal_multiplier * weekly_multiplier * year_multiplier
            daily_sales += np.random.normal(0, daily_sales * 0.15)
            daily_sales = max(10, int(daily_sales))
            
            for sale_id in range(min(daily_sales, 1000)):
                category = np.random.choice(book_categories, p=[0.3, 0.25, 0.25, 0.1, 0.05, 0.05])
                subject = np.random.choice(subjects, p=[0.25, 0.2, 0.15, 0.1, 0.1, 0.05, 0.1, 0.05])
                publisher = np.random.choice(publishers, p=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
                
                price_map = {'University Books': 180000, 'High School Books': 95000, 
                           'Reference Books': 95000, 'Middle School Books': 75000}
                base_price = price_map.get(category, 45000)
                
                publisher_multiplier = {'Erlangga': 1.2, 'Tiga Serangkai': 1.0, 'Yudhistira': 1.1,
                                      'Esis': 1.0, 'Grafindo': 0.9, 'Bumi Aksara': 0.95}[publisher]
                
                price = int(base_price * publisher_multiplier * np.random.uniform(0.8, 1.3))
                quantity = np.random.choice([1, 2, 3, 5, 10], p=[0.4, 0.3, 0.15, 0.1, 0.05])
                
                if seasonal_multiplier > 3:
                    quantity *= np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                
                customer_type = np.random.choice(['Individual Parent', 'School Bulk', 'Teacher', 'Student'], 
                                               p=[0.5, 0.2, 0.2, 0.1])
                
                if customer_type == 'School Bulk':
                    quantity *= np.random.randint(5, 25)
                    price *= 0.85
                
                sale_time = date.replace(hour=np.random.randint(8, 20), minute=np.random.randint(0, 60))
                
                data.append({
                    'sale_date': date.strftime('%d/%m/%Y'),
                    'sale_time': sale_time.strftime('%H:%M'),
                    'book_category': category,
                    'subject': subject,
                    'publisher': publisher,
                    'unit_price_idr': price,
                    'quantity_sold': quantity,
                    'total_revenue_idr': price * quantity,
                    'customer_type': customer_type,
                    'school_grade_level': np.random.choice(['K-6', '7-9', '10-12', 'University'], 
                                                         p=[0.35, 0.25, 0.25, 0.15]),
                    'is_new_curriculum': np.random.choice([True, False], p=[0.7, 0.3])
                })
        
        return pd.DataFrame(data)