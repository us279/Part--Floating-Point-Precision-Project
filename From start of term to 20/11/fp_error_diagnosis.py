import sys
import numpy as np
import pandas as pd

# Instantiate MachAr
machar = np.MachAr()

# Gather MachAr information
machar_info = {
    "Machine epsilon": machar.eps,
    "Maximum floating-point number": machar.xmax,
    "Minimum positive floating-point number": machar.xmin,
    "Base of the exponent": machar.ibeta,
    "Digits in mantissa": machar.it,
    "Digits in exponent": machar.iexp,
    "Number of bits in exponent": machar.machep
}

# Gather sys.float_info information
sys_info = {
    "Machine epsilon": sys.float_info.epsilon,
    "Maximum floating-point number": sys.float_info.max,
    "Minimum positive floating-point number": sys.float_info.min,
    "Digits in mantissa": sys.float_info.dig,
    "Maximum exponent": sys.float_info.max_exp,
    "Minimum exponent": sys.float_info.min_exp
}

# Gather numpy.finfo(float64) information
finfo_info = {
    "Machine epsilon": np.finfo(np.float64).eps,
    "Maximum floating-point number": np.finfo(np.float64).max,
    "Minimum positive floating-point number": np.finfo(np.float64).tiny,
    "Base of the exponent": np.finfo(np.float64).machar.ibeta,
    "Digits in mantissa": np.finfo(np.float64).machar.it,
    "Digits in exponent": np.finfo(np.float64).machar.iexp,
    "Number of bits in exponent": np.finfo(np.float64).machar.machep
}

# Create DataFrame
df = pd.DataFrame({
    "MachAr": machar_info,
    "sys.float_info": sys_info,
    "numpy.finfo(float64)": finfo_info
})

# Transpose DataFrame for better readability
df = df.T

# Display DataFrame
print(df)
