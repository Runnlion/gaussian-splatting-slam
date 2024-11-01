import math

# Change in X and Y
delta_x = -11.695 - (-9.304) 
delta_y = 1.219  - (-3.091 )

# Heading angle in radians
theta_radians = math.atan2(delta_y, delta_x)

# Convert angle to degrees
theta_degrees = math.degrees(theta_radians)
print(theta_degrees)