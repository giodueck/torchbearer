train = {
    # October 2025
    'T20KNA_20251003T174955': 'https://download.dataspace.copernicus.eu/odata/v1/Products(c9c0a608-1ed5-41ee-98bb-2346ff1502d2)/$value',
    'T20KNB_20251003T174955': 'https://download.dataspace.copernicus.eu/odata/v1/Products(81e3454b-cf13-493b-b4ef-a8370ee4900f)/$value',
    # September 2025
    'T20KPC_20250913T194708': 'https://download.dataspace.copernicus.eu/odata/v1/Products(44342712-af91-4a2c-b434-9fa5553684a4)/$value',
    # June 2025
    'T20KNA_20250620T173703': 'https://download.dataspace.copernicus.eu/odata/v1/Products(e4cebb62-b507-4e8a-86e7-63986518b673)/$value',
    'T20KNB_20250620T173703': 'https://download.dataspace.copernicus.eu/odata/v1/Products(93cdefe8-faca-437a-8088-6716e6cc9e02)/$value',
    # May 2025
    'T20KPC_20250513T174055': 'https://download.dataspace.copernicus.eu/odata/v1/Products(bc019cf0-2df8-425c-a7d7-a4f4aaaa969e)/$value',
    # January 2025
    'T20KNA_20250116T211859': 'https://download.dataspace.copernicus.eu/odata/v1/Products(6ba80331-e386-412f-af90-42673e5438d9)/$value',
    'T20KNB_20250116T211859': 'https://download.dataspace.copernicus.eu/odata/v1/Products(ab3c6dad-998b-4542-9f49-2ecf267729ae)/$value',
    'T20KPC_20250116T211859': 'https://download.dataspace.copernicus.eu/odata/v1/Products(e5305356-5b3c-4d2f-af1d-deb506044d7e)/$value',
    # October 2024
    'T20KNA_20241003T200351': 'https://download.dataspace.copernicus.eu/odata/v1/Products(4033655f-4ddf-4f0c-93aa-e5f9c13746ae)/$value',
    'T20KNB_20241003T200351': 'https://download.dataspace.copernicus.eu/odata/v1/Products(cc08a79a-3f8e-4894-99f3-6922ff6ca549)/$value',
    'T20KPC_20241020T181904': 'https://download.dataspace.copernicus.eu/odata/v1/Products(9fbca04e-8190-463a-a40d-bddb986248af)/$value',
}

test = {
    'T20KQV_20251020T174731': 'https://download.dataspace.copernicus.eu/odata/v1/Products(665ca9a2-9f9f-457b-95b7-4629d24dcde8)/$value', # Neuland
    'T20KRA_20251020T174731': 'https://download.dataspace.copernicus.eu/odata/v1/Products(1228c9c5-a508-45b6-8637-4e01e47ad35a)/$value', # Loma Plata and Filadelfia
    'T20KNC_20250913T194708': 'https://download.dataspace.copernicus.eu/odata/v1/Products(7327d41c-8b46-4814-9b84-cf67f865000c)/$value', # Western-most part of Medanos, not all included in training
}

neuland = {
    'T20KQV_20251020T174731': 'https://download.dataspace.copernicus.eu/odata/v1/Products(665ca9a2-9f9f-457b-95b7-4629d24dcde8)/$value', # Neuland
}

lp_fila = {
    'T20KRA_20251020T174731': 'https://download.dataspace.copernicus.eu/odata/v1/Products(1228c9c5-a508-45b6-8637-4e01e47ad35a)/$value', # Loma Plata and Filadelfia
}

medanos = {
    'T20KNC_20250913T194708': 'https://download.dataspace.copernicus.eu/odata/v1/Products(7327d41c-8b46-4814-9b84-cf67f865000c)/$value', # Western-most part of Medanos, not all included in training
}

PRODUCTS = {
    'train': train,
    'test': test,
    'neuland': neuland,
    'lp_fila': lp_fila,
    'medanos': medanos,
}
