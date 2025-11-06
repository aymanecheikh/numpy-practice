from pprint import pprint
import numpy as np
from numpy.random import uniform
np.random.seed(1)


class Finance:
    '''
    You track the closing prices of a tech stock for 20 days.
    
    Generate an array of simulated daily prices starting from £120 and
    fluctuating ±2% each day.
    
    Compute the 5-day rolling average prices.
    '''
    @property
    def simulated_daily_prices(self):
        daily_price_movements = uniform(
            low=-0.02,
            high=0.02,
            size=20
        )
        daily_price_movements[0] = 120
        
        for index, price in enumerate(daily_price_movements):
            if index != 0:
                daily_price_movements[index] = (
                    (daily_price_movements[index-1])
                    * (1 + daily_price_movements[index])
                )
        
        return daily_price_movements.cumprod()
    
    @property
    def simulated_daily_prices_v2(self):
        return 120 + np.cumprod(1 + uniform(-1.02, 1.02, 20))

    
    @property
    def rolling_average(self):
        return np.convolve(
            self.simulated_daily_prices,
            np.ones(5)/5,
            mode='valid'
        )


class Agriculture:
    '''
    You monitor plant height (in cm) of 6 different crops across 4 weeks.
    
    Simulate the dataset as a 6 by 4 array with weekly height increments
    between 1.5 cm and 5 cm.
    
    Find which crop had the largest total height increase over 4 weeks.
    '''
    @property
    def weekly_height_increments(self):
        weekly_height_increments = uniform(
            low=1.5,
            high=5,
            size=(6,4)
        )
        plant_height_growth = np.apply_along_axis(
            func1d=np.cumsum,
            axis=1,
            arr=weekly_height_increments
        )
        total_height_increases = (
            plant_height_growth[:,-1] - plant_height_growth[:,0]
        )
        return f'Crop ID with largest total height increase: {
            np.where(
                max(total_height_increases) == total_height_increases
            )[0][0]
            }'


class Energy:
    '''
    A solar farm records hourly energy output (kWh) for 7 days.
    
    Create a 7 by 24 array of simulated data where daytime hours (6 to 18)
    produce between 100 to 500 kWh and other hours produce 0 to 50 kWh.
    
    Find the day with the highest total production.
    '''
    def generate_energy_output(self, low: int, high: int, size: int):
        return uniform(
            low=low,
            high=high,
            size=size
        )
    
    @property
    def highest_total_energy_production(self):
        day = uniform(
            low=100,
            high=500,
            size=(7, 12)
        )
        night = np.random.uniform(
            low=0,
            high=50,
            size=(7, 12)
        )
        total_daily_output = np.apply_along_axis(
            func1d=np.sum,
            axis=1,
            arr=day+night
        )
        return f'Day ID with largest total energy output: {
            np.where(max(total_daily_output) == total_daily_output)[0][0]
            }'


class Healthcare:
    '''
    You measure 8 patients heart rates every 2 hours over a 24-hour period.
    
    Build a 8 by 12 array with integer values between 60 and 100.
    
    Compute the mean heart rate for each patient and identify who had the
    highest average.
    '''
    @property
    def mean_heart_rate_per_patient(self):
        heart_rates = np.array(
            uniform(
                low=60,
                high=100,
                size=(8, 12)
            ),
            dtype='int8'
        )
        return np.apply_along_axis(
            func1d=np.mean,
            axis=1,
            arr=heart_rates
        )
    
    @property
    def highest_average_heart_rate(self):
        average_heart_rates = self.mean_heart_rate_per_patient
        return f'Patient ID with highest average heart rate: {
            np.where(average_heart_rates.max() == average_heart_rates)[0][0]
        }'


class Transport:
    '''
    You manage 10 electric vehicle chargers and record their usage (kWh
    delivered) for 5 days.
    
    Construct a 10 by 5 array of random integers between 20 and 120.
    
    Find the total energy dispensed per day and per station.
    '''
    @property
    def ev_charger_usage_data(self):
        return uniform(
            low=20,
            high=120,
            size=(10, 5)
        )
    
    @property
    def total_energy_dispersed_by_day(self):
        return float(np.apply_along_axis(
            func1d=np.sum,
            axis=0,
            arr=self.ev_charger_usage_data
        ).sum())
    
    @property
    def total_energy_dispersed_by_station(self):
        return float(np.apply_along_axis(
            func1d=np.sum,
            axis=1,
            arr=self.ev_charger_usage_data
        ).sum())
    

class Retail:
    '''
    A store sells 4 products across 6 regions.
    
    Generate a 4 by 6 array of daily unit sales (between 10 and 200).
    
    Compute the top-selling product and the highest-performing region.
    '''
    @property
    def daily_unit_sales(self):
        return np.array(
            uniform(
                low=10,
                high=200,
                size=(4, 6)
            ),
            dtype='int16'
        )
    
    @property
    def top_selling_product(self):
        total_unit_sales_by_product = np.apply_along_axis(
            func1d=np.sum,
            axis=1,
            arr=self.daily_unit_sales
        )
        return f'Product ID Sold the Most: {
            np.where(
                total_unit_sales_by_product.max()
                == total_unit_sales_by_product
            )[0][0]
        }'
    
    @property
    def highest_performing_region(self):
        total_unit_sales_by_region = np.apply_along_axis(
            func1d=np.sum,
            axis=0,
            arr=self.daily_unit_sales
        )
        return f'Product ID Sold the Most: {
            np.where(
                total_unit_sales_by_region.max()
                == total_unit_sales_by_region
            )[0][0]
        }'


class Sports:
    '''
    Each of 5 basketball players has their points scored across 10 games.
    
    Create a 5 by 10 array of integers between 5 and 40.
    
    Determine the average points per player and identify any game where a
    player scored above 35.
    '''
    @property
    def basketball_points_by_player(self):
        return np.array(
            uniform(
                low=5,
                high=40,
                size=(5,10)
            ),
            dtype='int8'
        )
    
    @property
    def average_points_per_player(self):
        return np.apply_along_axis(
            func1d=np.mean,
            axis=1,
            arr=self.basketball_points_by_player
        )
    
    @property
    def score_threshold_met(self):
        data = self.basketball_points_by_player
        return f'Game ID(s) where a player scored above 35: {
            np.where(data > 35)[1]
        }'


class Aerospace:
    '''
    You log hourly fuel consumption (kg) for a 4-engine aircraft over a
    10-hour flight.
    
    Simulate a 4 by 10 array of values between 400 and 900.
    
    Find the engine with the most consistent (least variable) consumption.
    '''
    @property
    def fuel_consumption_data(self):
        return uniform(
            low=400,
            high=900,
            size=(4, 10)
        )
    
    @property
    def most_stable_engine(self):
        fuel_consumption_std = np.apply_along_axis(
            func1d=np.std,
            axis=1,
            arr=self.fuel_consumption_data
        )
        return f'Most stable engine (ID): {
            np.where(fuel_consumption_std.max() == fuel_consumption_std)[0][0]
        }'


class Logistics:
    '''
    Track stock levels of 6 items over 8 days.
    
    Start each item at 100 units and subtract random daily shipments between 5
    and 15 units.
    
    Compute when each item first dropped below 50 units.
    '''
    @property
    def stock_levels(self):
        random_daily_subtractions = uniform(
            low=5,
            high=15,
            size=(6, 8)
        )
        random_daily_subtractions[:,0] = 100
        for index_i, i in enumerate(random_daily_subtractions):
            for index_j, j in enumerate(i):
                if index_j != 0:
                    i[index_j] = i[index_j-1] - i[index_j]
        return random_daily_subtractions
    
    @property
    def inventory_threshold(self):
        data = self.stock_levels
        threshold_filter = np.vstack(np.where(data < 50))
        item_tracker = {}
        for index, i in enumerate(threshold_filter[0]):
            if i in item_tracker:
                continue
            item_tracker[int(i)] = int(threshold_filter[:,index][-1])

        return item_tracker


class WeatherScience:
    '''
    A climate scientist collects temperature data over a 5 by 5 region for 7
    days.
    
    Simulate a 7 by 5 by 5 array of daily readings between 10 °C and 35 °C.
    
    Find the warmest cell average temperature over the week.
    '''
    @property
    def temperatures(self):
        temperature_data = uniform(
            low=10,
            high=35,
            size=(7, 5, 5)
        )
        return temperature_data
    
    @property
    def warmest_cell_avg_temp(self):
        return np.apply_along_axis(
            func1d=np.max,
            axis=1,
            arr=np.apply_along_axis(
                func1d=np.max,
                axis=1,
                arr=self.temperatures
            )
        ).mean()


if __name__ == '__main__':
    f = Finance()
    ag = Agriculture()
    e = Energy()
    h = Healthcare()
    t = Transport()
    r = Retail()
    s = Sports()
    ae = Aerospace()
    l = Logistics()
    ws = WeatherScience()