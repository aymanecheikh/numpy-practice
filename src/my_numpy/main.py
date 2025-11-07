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
        return 120 + np.cumprod(1 + uniform(-1.02, 1.02, 20))

    
    @property
    def rolling_average(self):
        return np.convolve(
            self.simulated_daily_prices,
            np.ones(5)/5,
            mode='valid'
        )
    
    '''
    Simulate daily returns (%) of a mutual fund for 30 days between −1% and
    +1.5%.
    
    Use np.cumsum() to calculate the cumulative return curve starting from an
    initial value of 100.
    '''
    @property
    def mutual_fund_daily_returns(self):
        return uniform(-0.01, 0.015, 30)
    
    @property
    def cumulative_return_curve(self):
        return 100 + np.cumsum(self.mutual_fund_daily_returns)


class Agriculture:
    '''
    You monitor plant height (in cm) of 6 different crops across 4 weeks.
    
    Simulate the dataset as a 6 by 4 array with weekly height increments
    between 1.5 cm and 5 cm.
    
    Find which crop had the largest total height increase over 4 weeks.
    '''
    @property
    def weekly_plant_height_increments(self):
        return uniform(1.5, 5, (6, 4))
    
    @property
    def weekly_plant_height_progression(self):
        return np.cumsum(self.weekly_plant_height_increments, axis=1)
    
    @property
    def total_height_increases(self):
        return (
            self.weekly_plant_height_progression[:,-1]
            - self.weekly_plant_height_progression[:,0]
        )
    
    @property
    def plant_with_largest_total_height_increase(self):
        total_height_increases = self.total_height_increases
        return np.where(
            (
                np.max(total_height_increases)
                == total_height_increases
            )
        )[0][0]
    
    '''
    Each week, a crop grows by a factor between 1.01 to 1.10 depending on
    weather.

    Use np.cumprod() to simulate plant height progression over 16 weeks
    starting from 10 cm.
    '''
    @property
    def crop_growth_factors(self):
        return uniform(low=1.01, high=1.1, size=16)
    
    @property
    def crop_height_progression(self):
        return 10 * np.cumprod(self.crop_growth_factors)



class Energy:
    '''
    A solar farm records hourly energy output (kWh) for 7 days.
    
    Create a 7 by 24 array of simulated data where daytime hours (6 to 18)
    produce between 100 to 500 kWh and other hours produce 0 to 50 kWh.
    
    Find the day with the highest total production.
    '''
    
    @property
    def highest_total_energy_production(self):
        day = uniform(100,500,(7, 12))
        night = uniform(0,50,(7, 12))
        total_daily_output = np.sum(np.sum([day, night], axis=0), axis=0)
        return np.where(total_daily_output.max()==total_daily_output)[0][0]

    
    '''
    A solar panel loses 0.2 to 0.4% efficiency per month due to weathering.
    
    Simulate 5 years of performance loss and use np.cumprod() to estimate
    total efficiency after each month.
    '''
    @property
    def solar_panel_efficiency_loss(self):
        return uniform(low=-0.002, high=-0.004, size=60)
    
    @property
    def solar_panel_efficiency(self):
        return np.cumprod(1 + self.solar_panel_efficiency_loss)
    
    '''
    A factory consumes between 120 kWh and 250 kWh hourly for 48 hours.
    
    Use np.cumsum() to calculate the cumulative power usage over the two-day
    period.
    '''
    @property
    def energy_consumption(self):
        return uniform(120, 250, 48)
    
    @property
    def cumulative_power_usage(self):
        return np.cumsum(self.energy_consumption)


class Healthcare:
    '''
    You measure 8 patients heart rates every 2 hours over a 24-hour period.
    
    Build a 8 by 12 array with integer values between 60 and 100.
    
    Compute the mean heart rate for each patient and identify who had the
    highest average.
    '''
    @property
    def highest_average_heart_rate(self):
        heart_rates = uniform(60,100,(8, 12))
        average_heart_rate_per_patient = np.mean(heart_rates, axis=0)
        return np.where(
            average_heart_rate_per_patient.max()
            ==average_heart_rate_per_patient
        )[0][0]
    
    '''
    A patient’s viral load doubles daily for the first 10 days, but random
    medication effectiveness reduces the growth factor to between 1.5 and 2.0.
    
    Simulate the viral load trajectory using np.cumprod().
    '''
    @property
    def viral_load_growth_factor(self):
        return uniform(low=1.5, high=2, size=10)
    
    @property 
    def viral_growth_trajectory(self):
        return np.cumprod(self.viral_load_growth_factor)


class Transport:
    '''
    You manage 10 electric vehicle chargers and record their usage (kWh
    delivered) for 5 days.
    
    Construct a 10 by 5 array of random integers between 20 and 120.
    
    Find the total energy dispensed per day and per station.
    '''
    @property
    def ev_charger_usage_data(self):
        return np.array(
            uniform(
                low=20,
                high=120,
                size=(10, 5)
            ),
            dtype='int'
        )
    
    @property
    def total_energy_dispersed_by_day(self):
        return np.sum(self.ev_charger_usage_data, axis=1).sum()
    
    @property
    def total_energy_dispersed_by_station(self):
        return np.sum(self.ev_charger_usage_data, axis=0).sum()
    
    '''
    Each station adds a random multiplier between 1.00 and 1.05 to the current
    delay time due to compounding congestion.
    
    Use np.cumprod() to simulate total delay accumulation over 20 stations
    starting from a 2-minute base delay.
    '''
    @property
    def compounding_congestion_multiplier(self):
        return uniform(low=1, high=1.05, size=20)
    
    @property
    def total_delay_accumulations(self):
        return 2 * np.cumprod(self.compounding_congestion_multiplier)
    
    '''
    A delivery van travels random distances between 8 km and 15 km per trip
    across 25 trips.
    
    Use np.cumsum() to find the total distance covered after each trip.
    '''
    @property
    def distances_per_trip(self):
        return uniform(8, 15, 25)
    
    @property
    def total_distance(self):
        return np.cumsum(self.distances_per_trip)
    

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
            dtype='int'
        )
    
    @property
    def top_selling_product(self):
        total_sales_by_product =  np.sum(self.daily_unit_sales, axis=1)
        return np.where(
            total_sales_by_product.max()==total_sales_by_product
        )[0][0]
    
    @property
    def highest_performing_region(self):
        total_sales_by_region =  np.sum(self.daily_unit_sales, axis=0)
        print(total_sales_by_region)
        return np.where(
            total_sales_by_region.max()==total_sales_by_region
        )[0][0]
    
    '''
    A store retains 95 to 99% of its customers each month.
    
    Generate 24 months of random retention rates and use cumulative product to
    estimate the fraction of original customers remaining each month.
    '''
    @property
    def retention_rates(self):
        data = np.cumprod(uniform(low=0.95, high=0.99, size=24))
        return data


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
    
    '''
    An athlete’s performance metric is multiplied each game by a random
    consistency factor between 0.95 and 1.05.
    
    Simulate 30 games and use np.cumprod() to visualize how small performance
    fluctuations accumulate over a season.
    '''
    @property
    def athlete_performance_metric(self):
        return uniform(low=0.95, high=1.05, size=30)
    
    @property
    def performance_flucutation_accumulation(self):
        return np.cumprod(self.athlete_performance_metric)


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
    
    '''
    A spacecraft 's fuel efficiency decreases by 0.3% with each kilometer
    ascended due to gravity and drag.
    
    Generate a 10-element array representing efficiency multipliers and
    compute the cumulative product to estimate efficiency after each
    kilometer.
    '''
    @property
    def spacecraft_fuel_efficiency_decline(self):
        return np.cumprod(1 + np.full(shape=10, fill_value=-0.003))



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


class ClimateScience:
    '''
    A climate scientist collects temperature data over a 5 by 5 region for 7
    days.
    
    Simulate a 7 by 5 by 5 array of daily readings between 10 °C and 35 °C.
    
    Find the warmest cell average temperature over the week.
    '''
    @property
    def temperatures(self):
        return uniform(low=10,high=35,size=(7, 5, 5))
    
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
    
    '''
    A model predicts that each month’s temperature deviation is 0.1 to 0.3
    times the previous month’s anomaly due to climate inertia.
    
    Use np.cumprod() to simulate 24 months of compounding anomaly effects
    starting from a baseline deviation of +1 °C.
    '''
    @property
    def temperature_deviation(self):
        return uniform(low=0.1, high=0.3, size=24)
    
    @property
    def compounding_anomaly_effects(self):
        return 1 + np.cumprod(1 + self.temperature_deviation)


class Pharmaceuticals:
    '''
    Simulate hourly absorption rates (as percentages) for a new oral drug over
    12 hours.
    
    Use np.cumprod() to estimate total absorption progression over time.
    '''
    @property
    def hourly_absorbtion_rates(self):
        return uniform(0.95, 0.99, 12)
    
    @property
    def total_absorbtion_progression(self):
        return np.cumprod(self.hourly_absorbtion_rates)


class Manufacturing:
    '''
    Each production stage has a yield rate between 93 to 99%.
    
    Generate 15 stages and use np.cumprod() to estimate total yield across the
    full process.
    '''
    @property
    def production_stage_yield_rates(self):
        return uniform(low=0.93, high=0.99, size=15)
    
    @property
    def total_production_yield(self):
        return np.cumprod(self.production_stage_yield_rates)

    


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
    cs = ClimateScience()
    p = Pharmaceuticals()
    m = Manufacturing()