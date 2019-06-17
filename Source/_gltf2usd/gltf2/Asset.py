class Asset(object):
    def __init__(self, asset_entry):
        self._generator = asset_entry['generator'] if 'generator' in asset_entry else None
        self._version = asset_entry['version'] if 'version' in asset_entry else None
        self._extras = asset_entry['extras'] if 'extras' in asset_entry else {}
        self._copyright = asset_entry['copyright'] if 'copyright' in asset_entry else None
        self._minversion = asset_entry['minversion'] if 'minversion' in asset_entry else None

    @property
    def generator(self):
        return self._generator

    @property
    def version(self):
        return self._version

    @property
    def minversion(self):
        return self._minversion

    @property
    def copyright(self):
        return self._copyright

    @property
    def extras(self):
        return self._extras

