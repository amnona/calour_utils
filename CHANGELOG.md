# calour_utils changelog

## 2025.12.04
Bug fixes:
* Update the variance_stat score (used in group_dependence) to 0-(within groups / total) instead of (total/within groups) to fix the divide by zero
* Remove the skip_filter parameter from group_dependence, and instead use the exp.negatives parameter to skip filtering if True
