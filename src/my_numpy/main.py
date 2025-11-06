import numpy as np


class Finance:
    '''
    You track the closing prices of a tech stock for 20 days.
    
    Generate an array of simulated daily prices starting from £120 and
    fluctuating ±2% each day.
    
    Compute the 5-day rolling average prices.
    '''
    @property
    def simulated_daily_prices(self):
        daily_price_movements = np.random.uniform(
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
        return daily_price_movements
    
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
    
    Simulate the dataset as a 6 by 4 array with weekly height increments between
    1.5 cm and 5 cm.
    
    Find which crop had the largest total height increase over 4 weeks.
    '''
    @property
    def weekly_height_increments(self):
        weekly_height_increments = np.random.uniform(
            low=1.5,
            high=5,
            size=(6,4)
        )
        plant_height_growth = np.apply_along_axis(
            func1d=np.cumsum,
            axis=1,
            arr=weekly_height_increments
        )
        total_height_increases = plant_height_growth[:,-1] - plant_height_growth[:,0]
        return f'Crop ID with largest total height increase: {
            np.where(max(total_height_increases) == total_height_increases)[0]
            }'


if __name__ == '__main__':
    f = Finance()
    a = Agriculture()