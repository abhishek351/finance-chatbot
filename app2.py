
def monkeys_left(n, k, j, m, p):
  """
  Calculates the number of monkeys left on a tree after eating bananas and peanuts.

  Args:
    n: Total number of monkeys.
    k: Number of bananas a single monkey can eat.
    j: Number of peanuts a single monkey can eat.
    m: Total number of bananas available.
    p: Total number of peanuts available.

  Returns:
    A string with the number of monkeys left on the tree.

  Raises:
    ValueError: If any input is invalid or not a positive integer.
  """

  # Validate inputs
  if not isinstance(n, int) or not isinstance(k, int) or not isinstance(j, int) or \
      not isinstance(m, int) or not isinstance(p, int) or n < 1 or k < 1 or j < 1:
    raise ValueError("INVALID INPUT")

  # Count monkeys that can eat all bananas or peanuts
  monkeys_ate_all = min(m // k + p // j, n)
  n -= monkeys_ate_all

  # Return message with remaining monkeys
  return f"Number of Monkeys left on the Tree:{n}"

# Example usage
n, k, j, m, p = 20, 2, 3, 12, 12
print(monkeys_left(n, k, j, m, p))