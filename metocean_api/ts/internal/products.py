from .tudelft import tudelft_products as tudelft
from .metno import met_products as metno
from .ec import ec_products as ec
from .product import Product

def find_product(name: str) -> Product:
    """Find the product by name"""
    for module in [metno, ec, tudelft]:
        produkt = module.find_product(name)
        if produkt is not None:
            return produkt
    raise ValueError(f"Product not recognized {name}")
