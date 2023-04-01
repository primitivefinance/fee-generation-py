import numpy as np
import requests
import json
import time

# Randomly Generated Price Paths

def generateGBM(T, mu, sigma, p0, dt, env):
            N = round(T/dt)
            t = dt*env.now
            dW = np.random.standard_normal(size=N)
            W = np.cumsum(dW)[int(env.now)]*np.sqrt(dt)
            P = p0*np.exp((mu - 0.5*sigma**2)*t + sigma*W)
            return P

def generateOU(T, mean, sigma, p0, dt, theta, env):
    N = round(T/dt)
    dW = np.random.standard_normal(size=N)
    W = np.cumsum(dW)[int(env.now)] * np.sqrt(dt)
    t = dt * env.now
    term = np.exp(-theta * t)
    OU = p0 * term + mean * (1 - term) + sigma * np.sqrt(dt / (2 * theta)) * W
    return OU
      
# Backtesting Data USDC/USDT

def queryUniswapV3(query, max_retries=5, initial_wait_time=1):
    url = 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3'
    headers = {'Content-Type': 'application/json'}
    
    for retry in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json={'query': query})
            response_json = response.json()
            
            if 'errors' in response_json:
                raise ValueError(f"Error in query response: {response_json['errors']}")

            return response_json['data']
        except requests.exceptions.ConnectionError as e:
            if retry == max_retries - 1:
                raise e
            wait_time = initial_wait_time * (2 ** retry)
            time.sleep(wait_time)

def fetch_all_hourly_data(pool_id, start_timestamp, end_timestamp):
    all_data = []
    chunk_size = 1000

    current_timestamp = start_timestamp

    while True:
        query = f'''
        {{
            pool(id: "{pool_id}") {{
                poolHourData(first: {chunk_size}, where: {{ periodStartUnix_gt: {current_timestamp}, periodStartUnix_lt: {end_timestamp} }}) {{
                    periodStartUnix
                    high
                    low
                    open
                    close
                }}
            }}
        }}
        '''
        response_data = queryUniswapV3(query)
        response = response_data.get('pool', {}).get('poolHourData')

        if not response:  # Break the loop if no more data is available
            break

        all_data.extend(response)

        current_timestamp = int(response[-1]['periodStartUnix'])

    return all_data

pool_id = "0x3416cf6c708da44db2624d63ea0aaef7113527c6"      # USDC/USDT
start_timestamp = 1648512000                                # March 29th, 2022 00:00:00 UTC
end_timestamp = 1670073600                                  # September 30th, 2022 00:00:00 UTC

all_hourly_data = fetch_all_hourly_data(pool_id, start_timestamp, end_timestamp)
close_values = [float(hour_data["close"]) for hour_data in all_hourly_data]