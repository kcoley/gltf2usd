class Version(object):
    """Version API for gltf2usd
    """
    _major = 0
    _minor = 1
    _patch = 15
    @staticmethod
    def get_major_version_number():
        """Returns the major version
        
        Returns:
            int -- major version number
        """

        return Version._major
    
    @staticmethod
    def get_minor_version_number():
        """Returns the minor version number
        
        Returns:
            int -- minor version number
        """

        return Version._minor

    @staticmethod
    def get_patch_version_number():
        """Patch version number
        
        Returns:
            int -- patch version number
        """

        return Version._patch

    @staticmethod
    def get_version_name():
        """Returns the version name
        
        Returns:
            str -- version mame
        """

        return '{0}.{1}.{2}'.format(Version._major, Version._minor, Version._patch)