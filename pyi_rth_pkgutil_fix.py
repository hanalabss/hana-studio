# Python 3.12 호환성을 위한 pkgutil runtime hook 수정
def _pyi_rthook():
    import pkgutil
    try:
        import pyimod02_importers

        # Python 3.12에서는 PyiFrozenFinder 대신 PyiFrozenLoader 사용
        if hasattr(pyimod02_importers, 'PyiFrozenFinder'):
            finder_class = pyimod02_importers.PyiFrozenFinder
        elif hasattr(pyimod02_importers, 'PyiFrozenLoader'):
            finder_class = pyimod02_importers.PyiFrozenLoader
        else:
            # 둘 다 없으면 패스
            return

        def _iter_pyi_frozen_finder_modules(finder, prefix=''):
            # Fetch PYZ TOC tree
            pyz_toc_tree = pyimod02_importers.get_pyz_toc_tree()

            if hasattr(finder, '_pyz_entry_prefix') and finder._pyz_entry_prefix:
                pkg_name_parts = finder._pyz_entry_prefix.split('.')
            else:
                pkg_name_parts = []

            tree_node = pyz_toc_tree
            for pkg_name_part in pkg_name_parts:
                tree_node = tree_node.get(pkg_name_part)
                if not isinstance(tree_node, dict):
                    tree_node = {}
                    break

            for entry_name, entry_data in tree_node.items():
                is_pkg = isinstance(entry_data, dict)
                yield prefix + entry_name, is_pkg

            if hasattr(finder, 'fallback_finder') and finder.fallback_finder is not None:
                yield from pkgutil.iter_importer_modules(finder.fallback_finder, prefix)

        pkgutil.iter_importer_modules.register(
            finder_class,
            _iter_pyi_frozen_finder_modules,
        )
    except Exception:
        # 오류가 발생하면 무시 (기본 동작 사용)
        pass

_pyi_rthook()
del _pyi_rthook
