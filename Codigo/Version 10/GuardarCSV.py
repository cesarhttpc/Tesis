# %%
import numpy as np
import pandas as pd

# Example arrays
array1 = np.array([1, 2, 3])
array2 = np.array([7, 8, 9])

# Convert arrays to DataFrames
df1 = pd.DataFrame(array1)
df2 = pd.DataFrame(array2)

print(df2)
# %%

# Open a CSV file in write mode
with open("multiple_arrays_pandas.csv", "w") as file:
    # Write the first DataFrame
    df1.to_csv(file, header=False, index=False)
    
    # Add a blank line (or any other separator) between DataFrames
    file.write("\n")
    
    # Write the second DataFrame
    df2.to_csv(file, header=False, index=False)

# # %%
# np.savetxt('values.csv', narr, delimiter=",")


# %%
import numpy as np

# Example arrays
array1 = np.array([[1, 2, 3], [4, 5, 6]]).T
array2 = np.array([[7, 8, 9], [10, 11, 12]]).T
print(array1)
print(array2)

# Open a file in write mode
with open("multiple_arrays.csv", "w") as file:
    # Save the first array
    np.savetxt(file, array1, delimiter=",", fmt='%d')
    
    # Add a blank line (or any other separator) between arrays
    file.write("\n")
    
    # Save the second array
    np.savetxt(file, array2, delimiter=",", fmt='%d')


# %%

import numpy as np

# Example arrays
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
array3 = np.array([7, 8, 9])
print(array1)

# Ensure all arrays are column vectors
array1 = array1.reshape(-1, 1)
array2 = array2.reshape(-1, 1)
array3 = array3.reshape(-1, 1)
print(array1)

# Concatenate arrays horizontally
combined_array = np.hstack((array1, array2, array3))

# Save the combined array to a CSV file
np.savetxt("combined_arrays.csv", combined_array, delimiter=",", fmt='%d')

print("Arrays saved to combined_arrays.csv")
