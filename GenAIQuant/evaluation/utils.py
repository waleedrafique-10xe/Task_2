def _create_kv_string(key: str, value):
    """
    Create a formatted key-value string for display purposes.

    Args:
        key (str): The key name to display
        value: The value to format. Can be:
            - A list/tuple of numeric values (formatted to 3 decimal places)
            - A single numeric value (formatted to 3 decimal places)
            - Any other type (converted to string as-is)

    Returns:
        str: A formatted string in the form "key: value" or "key: [val1, val2, ...]"

    Examples:
        >>> _create_kv_string("accuracy", 0.95678)
        'accuracy: 0.957'
        >>> _create_kv_string("losses", [0.1234, 0.5678])
        'losses: [0.123, 0.568]'
        >>> _create_kv_string("status", "complete")
        'status: complete'
    """
    if isinstance(value, (list, tuple)):
        # Handle lists/tuples of values
        formatted_values = []
        for x in value:
            if isinstance(x, (int, float)):
                formatted_values.append(f"{x:.3f}")
            else:
                formatted_values.append(str(x))
        output_string = f"{key}: [{', '.join(formatted_values)}]"
    elif isinstance(value, (int, float)):
        # Handle single numeric values
        output_string = f"{key}: {value:.3f}"
    else:
        # Handle all other types (strings, bools, None, etc.)
        output_string = f"{key}: {value}"

    return output_string


def print_results_table(data):
    # Flatten data and prepare all metric lines upfront
    rows = []
    for task, subsets in data.items():
        for subset, metrics in subsets.items():
            metric_lines = [
                _create_kv_string(k, v) for k, v in metrics.items() if k != "alias"
            ]
            rows.append((task, subset, metric_lines))

    if not rows:
        print("No data to display")
        return

    # Compute column widths with padding
    task_w = max(len(str(r[0])) for r in rows)
    subset_w = max(len(str(r[1])) for r in rows)
    metric_w = max(len(line) for _, _, lines in rows for line in lines)

    # Ensure minimum width for headers
    task_w = max(task_w, len("Task"))
    subset_w = max(subset_w, len("Subset"))
    metric_w = max(metric_w, len("Metrics"))

    # Border components
    def separator():
        return f"+{'-' * (task_w + 2)}+{'-' * (subset_w + 2)}+{'-' * (metric_w + 2)}+"

    def section_separator():
        return f"+{'=' * (task_w + 2)}+{'=' * (subset_w + 2)}+{'=' * (metric_w + 2)}+"

    # Print header
    print()
    print(section_separator())
    print(
        f"| {'Task'.ljust(task_w)} | {'Subset'.ljust(subset_w)} |"
        f" {'Metrics'.ljust(metric_w)} |"
    )
    print(section_separator())

    # Print data rows
    last_task = None
    for idx, (task, subset, metric_lines) in enumerate(rows):
        # Add thick separator between different tasks
        if last_task is not None and task != last_task:
            print(section_separator())

        show_task = last_task is None or task != last_task
        task_display = task if show_task else ""

        if metric_lines:
            print(
                f"| {task_display.ljust(task_w)} | {subset.ljust(subset_w)} |"
                f" {metric_lines[0].ljust(metric_w)} |"
            )

            for metric_line in metric_lines[1:]:
                print(
                    f"| {''.ljust(task_w)} | {''.ljust(subset_w)} |"
                    f" {metric_line.ljust(metric_w)} |"
                )

            next_idx = idx + 1
            if next_idx < len(rows):
                next_task = rows[next_idx][0]
                if next_task == task:  # Same task, just separate subsets
                    print(separator())
        else:
            print(
                f"| {task_display.ljust(task_w)} | {subset.ljust(subset_w)} |"
                f" {'(no metrics)'.ljust(metric_w)} |"
            )
            if idx + 1 < len(rows) and rows[idx + 1][0] == task:
                print(separator())

        last_task = task

    print(section_separator())
    print()
