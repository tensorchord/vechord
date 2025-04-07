# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "vechord"
copyright = "2025, TensorChord"
author = "TensorChord"
release = "latest"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinxext.opengraph",
    "sphinx_sitemap",
]

templates_path = ["_templates"]
exclude_patterns = []
source_suffix = [".rst", ".md"]

# Extensions
myst_heading_anchors = 3
autodoc_member_order = "bysource"
# napoleon
napoleon_attr_annotations = True
napoleon_include_init_with_doc = True
napoleon_use_admonition_for_references = True
# opengraph
ogp_site_url = "https://github.com/tensorchord/vechord"
ogp_image = (
    "https://github.com/user-attachments/assets/7b2819bb-1a7d-4b84-9ff9-d0c4d5340da9"
)
# sitemap
html_baseurl = "https://tensorchord.github.io/vechord/"
html_extra_path = ["robots.txt"]
# myst
myst_enable_extensions = ["tasklist", "fieldlist", "colon_fence"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_logo = (
    "https://github.com/user-attachments/assets/7b2819bb-1a7d-4b84-9ff9-d0c4d5340da9"
)
html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/tensorchord/vechord",
    "source_branch": "main",
    "source_directory": "docs/source",
}
