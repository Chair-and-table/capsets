import numpy as np

def is_cap_set(vectors: np.ndarray) -> bool:
  """Returns whether `vectors` form a valid cap set.

  Checking the cap set property naively takes O(c^3 n) time, where c is the size
  of the cap set. This function implements a faster check that runs in O(c^2 n).

  Args:
    vectors: [c, n] array containing c n-dimensional vectors over {0, 1, 2}.
  """
  _, n = vectors.shape

  # Convert `vectors` elements into raveled indices (numbers in [0, 3^n) ).
  powers = np.array([3 ** j for j in range(n - 1, -1, -1)], dtype=int)  # [n]
  raveled = np.einsum('in,n->i', vectors, powers)  # [c]

  # Starting from the empty set, we iterate through `vectors` one by one and at
  # each step check that the vector can be inserted into the set without
  # violating the defining property of cap set. To make this check fast we
  # maintain a vector `is_blocked` indicating for each element of Z_3^n whether
  # that element can be inserted into the growing set without violating the cap
  # set property.
  is_blocked = np.full(shape=3 ** n, fill_value=False, dtype=bool)
  for i, (new_vector, new_index) in enumerate(zip(vectors, raveled)):
    if is_blocked[new_index]:
      return False  # Inserting the i-th element violated the cap set property.
    if i >= 1:
      # Update which elements are blocked after the insertion of `new_vector`.
      blocking = np.einsum(
          'nk,k->n',
          (- vectors[:i, :] - new_vector[None, :]) % 3, powers)
      is_blocked[blocking] = True
    is_blocked[new_index] = True  # In case `vectors` contains duplicates.
  return True  # All elements inserted without violating the cap set property.