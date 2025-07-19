import numpy as np
from fredapi import Fred
import torch
import time


fred = Fred(api_key='c496761920d5eedc780bf493ac3ae0e9')
ten_year_treasury_note = fred.get_series_latest_release('GS10') / 100
risk_free_rate = ten_year_treasury_note.iloc[-1]

def JR_Method_GPU(strike, time_to_maturity, initial_price, rate, time_step, sigma, option_type='P', batch_size=1):
    if not isinstance(strike, torch.Tensor):
        strike = torch.tensor([strike] * batch_size, dtype=torch.float32).cuda()
    if not isinstance(initial_price, torch.Tensor):
        initial_price = torch.tensor([initial_price] * batch_size, dtype=torch.float32).cuda()
    if not isinstance(rate, torch.Tensor):
        rate = torch.tensor([rate] * batch_size, dtype=torch.float32).cuda()
    if not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor([sigma] * batch_size, dtype=torch.float32).cuda()

    # Precompute values
    dt = time_to_maturity / time_step
    nu = rate - 0.5 * sigma ** 2
    up = torch.exp(nu * dt + sigma * torch.sqrt(torch.tensor(dt).cuda()))
    down = 1 / up
    q = (torch.exp(rate * dt) - down) / (up - down)
    disc = torch.exp(-rate * dt)

    # Create index tensors for vectorized operations
    steps = torch.arange(time_step, -1, -1).cuda()
    steps_up = torch.arange(0, time_step + 1, 1).cuda()

    # Compute all possible stock prices at maturity (vectorized)
    # Add batch dimension
    down_powers = down.unsqueeze(1) ** steps.unsqueeze(0)
    up_powers = up.unsqueeze(1) ** steps_up.unsqueeze(0)
    S = initial_price.unsqueeze(1) * down_powers * up_powers

    # Initialize option payoff
    if option_type == 'P':
        C = torch.maximum(strike.unsqueeze(1) - S, torch.tensor(0.0).cuda())
    else:
        C = torch.maximum(S - strike.unsqueeze(1), torch.tensor(0.0).cuda())

    # Backward recursion through the tree
    for i in range(time_step - 1, -1, -1):
        # Compute stock prices for current step
        steps_i = torch.arange(i, -1, -1).cuda()
        steps_up_i = torch.arange(0, i + 1, 1).cuda()
        down_powers = down.unsqueeze(1) ** steps_i.unsqueeze(0)
        up_powers = up.unsqueeze(1) ** steps_up_i.unsqueeze(0)
        S = initial_price.unsqueeze(1) * down_powers * up_powers

        # Option value backpropagation
        C_next = C[:, 1:i + 2]
        C_current = C[:, 0:i + 1]
        C = C[:, :i + 1]
        C = disc.unsqueeze(1) * (q.unsqueeze(1) * C_next + (1 - q).unsqueeze(1) * C_current)

        # Early exercise condition
        if option_type == 'P':
            C = torch.maximum(C, strike.unsqueeze(1) - S)
        else:
            C = torch.maximum(C, S - strike.unsqueeze(1))

    return C[:, 0]

def JR_Method(strike, time_to_maturity, initial_price, rate, time_step, sigma, option_type='P'):
    # precompute values
    dt = time_to_maturity / time_step
    nu = rate - 0.5 * sigma ** 2
    up = np.exp(nu * dt + sigma * np.sqrt(dt))
    down = 1 / up
    q = (np.exp(rate * dt) - down) / (up - down)
    disc = np.exp(-rate * dt)

    # initialise stock prices at maturity
    S = initial_price * down ** (np.arange(time_step, -1, -1)) * up ** (np.arange(0, time_step + 1, 1))

    # option payoff
    if option_type == 'P':
        C = np.maximum(0, strike - S)
    else:
        C = np.maximum(0, S - strike)

    # backward recursion through the tree
    for i in np.arange(time_step - 1, -1, -1):
        S = initial_price * down ** (np.arange(i, -1, -1)) * up ** (np.arange(0, i + 1, 1))
        C[:i + 1] = disc * (q * C[1:i + 2] + (1 - q) * C[0:i + 1])
        C = C[:-1]
        if option_type == 'P':
            C = np.maximum(C, strike - S)
        else:
            C = np.maximum(C, S - strike)

    return C[0]


# Initialise parameters
S0 = 258 # initial stock price
K = 260  # strike price
T = 3/365  # time to maturity in years
r = risk_free_rate  # annual risk-free rate
N = 40000  # number of time steps
annualized_volatility = 0.13

start = time.time()
call_option_price = JR_Method(K, T, S0, r, N, sigma=annualized_volatility, option_type='C')
end = time.time() - start
print(f"End time for vectorized CPU computation: {end:.2f}s")
print(f"Call Price: ${call_option_price:.2f}")

start1 = time.time()
gpu_price = JR_Method_GPU(K, T, S0, r, N, sigma=annualized_volatility, option_type='C', batch_size=1)
end1 = time.time() - start1
print(f"End time for GPU computation: {end1:.2f}s")
print(f"Call Price: ${gpu_price[0]:.2f}")


