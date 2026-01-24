"""
Explainability module for XAI analysis using SHAP and LIME.
"""
from explainability.explainers import (
    create_shap_explainer,
    compute_shap_values,
    plot_shap_summary,
    plot_shap_bar,
    plot_shap_waterfall,
    create_lime_explainer,
    explain_instance_lime,
    plot_lime_explanation,
    plot_lime_multiple_instances,
    get_representative_instances
)

__all__ = [
    'create_shap_explainer',
    'compute_shap_values',
    'plot_shap_summary',
    'plot_shap_bar',
    'plot_shap_waterfall',
    'create_lime_explainer',
    'explain_instance_lime',
    'plot_lime_explanation',
    'plot_lime_multiple_instances',
    'get_representative_instances'
]
