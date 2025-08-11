def scale_profile_points(points, bowl_radius, foot_radius, height):
    """Scales a normalized profile to the user's dimensions, ensuring x is increasing."""
    x_norm, y_norm = points[:, 0], points[:, 1]

    # --- FIX STARTS HERE ---
    # 1. Sort the (x, y) pairs based on the x-coordinates to ensure they are in order.
    # This maintains the integrity of the shape while satisfying the interpolator.
    sort_indices = np.argsort(x_norm)
    x_norm_sorted = x_norm[sort_indices]
    y_norm_sorted = y_norm[sort_indices]

    # 2. Scale the now-sorted points to the new dimensions.
    x_scaled = foot_radius + (x_norm_sorted - x_norm_sorted.min()) * (bowl_radius - foot_radius)
    y_scaled = y_norm_sorted * height

    # 3. As a final safeguard, remove any duplicate x-values that could cause issues.
    # np.unique also guarantees the output is sorted and strictly increasing.
    unique_x_scaled, unique_indices = np.unique(x_scaled, return_index=True)
    unique_y_scaled = y_scaled[unique_indices]
    # --- FIX ENDS HERE ---
    
    # Check if there are enough points to interpolate
    if len(unique_x_scaled) < 2:
        # Cannot interpolate with less than 2 points, return a straight line
        return unique_x_scaled, unique_y_scaled

    # Proceed with interpolation using the cleaned and sorted data
    interp_func = PchipInterpolator(unique_x_scaled, unique_y_scaled)
    x_smooth = np.linspace(unique_x_scaled.min(), unique_x_scaled.max(), 200)
    y_smooth = interp_func(x_smooth)
    
    return x_smooth, y_smooth
