def print_variables_of_class(class_name):
    """
    Print user-defined variables of a class.

    Args:
        class_name: The class object whose variables are to be printed.
    """
    # Get all variables of the class
    variables = vars(class_name)

    # Filter out special attributes
    user_defined_variables = {key: value for key, value in variables.items() 
                            if not key.startswith("__") and not callable(value)}

    # Print user-defined variables
    for var_name, var_value in user_defined_variables.items():
        print(f"{var_name}: {var_value}")