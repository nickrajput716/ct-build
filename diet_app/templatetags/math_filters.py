"""
Custom Template Filters for Mathematical Operations
Place this file at: diet_app/templatetags/math_filters.py

Directory structure should be:
diet_app/
    templatetags/
        __init__.py  (empty file, must exist!)
        math_filters.py  (this file)
"""

from django import template

register = template.Library()


@register.filter(name='multiply')
def multiply(value, arg):
    """
    Multiply the value by the argument
    Usage in template: {{ value|multiply:arg }}
    """
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0


@register.filter(name='divide')
def divide(value, arg):
    """
    Divide the value by the argument
    Usage in template: {{ value|divide:arg }}
    """
    try:
        return float(value) / float(arg)
    except (ValueError, TypeError, ZeroDivisionError):
        return 0


@register.filter(name='subtract')
def subtract(value, arg):
    """
    Subtract the argument from the value
    Usage in template: {{ value|subtract:arg }}
    """
    try:
        return float(value) - float(arg)
    except (ValueError, TypeError):
        return 0


@register.filter(name='add_num')
def add_num(value, arg):
    """
    Add the argument to the value
    Usage in template: {{ value|add_num:arg }}
    Note: Django has built-in 'add' filter, but this ensures numeric addition
    """
    try:
        return float(value) + float(arg)
    except (ValueError, TypeError):
        return 0


@register.filter(name='percentage')
def percentage(value, arg):
    """
    Calculate percentage of value
    Usage in template: {{ value|percentage:20 }} gives 20% of value
    """
    try:
        return (float(value) * float(arg)) / 100
    except (ValueError, TypeError, ZeroDivisionError):
        return 0


@register.filter(name='abs_value')
def abs_value(value):
    """
    Return absolute value
    Usage in template: {{ value|abs_value }}
    """
    try:
        return abs(float(value))
    except (ValueError, TypeError):
        return 0