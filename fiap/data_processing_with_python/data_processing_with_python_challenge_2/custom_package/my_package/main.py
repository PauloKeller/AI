def calculate_average(scores: list[int] = []):
    """
    This function takes a list of numbers as student's tests score and returns the average of those.

    Parameters:
    score (list[int]): The student's scores.

    Returns:
    int: The sum of scores and divided by the amount (size) of scores.
    """

    result = sum(scores) / len(scores)
    return result