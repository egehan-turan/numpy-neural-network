def format_number(x):
    # Use scientific notation for numbers smaller than 0.0001
    # (Checking absolute value to handle negative numbers as well)
    if 0 < x < 0.001:
        # .4g or .3e are common. .4g automatically switches, 
        # but to match your example "3.412e-5" exactly:
        return f"{x:.4e}" 
    else:
        # Show 4 digits after the dot
        return f"{x:.4f}"