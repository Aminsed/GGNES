"""
Serialization & Migration Tutorial

Goals:
- Demonstrate consolidated report migration to v2 and strict validation
"""

from ggnes.utils.observability import (
    migrate_consolidated_report_to_v2,
    validate_consolidated_report_v2,
)


def main():
    v1 = {
        'derivation_checksum': 'abc',
        'wl_fingerprint': 'def',
        'batches': [],
        'determinism_checksum': 'ghi',
    }
    v2 = migrate_consolidated_report_to_v2(v1)
    validate_consolidated_report_v2(v2)
    print('schema_version:', v2['schema_version'])


if __name__ == '__main__':
    main()


