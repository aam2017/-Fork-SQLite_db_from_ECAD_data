# -*- coding: utf-8 -*-

def dms_to_dec(dms):
    """Convert latitude/longitude from degrees:minutes:seconds to decimal."""
    # Remove colon separators and split:
    dms = dms.replace(':',' ').split()
    # There may be leading zeros in the string. Strip only leading (not
    # trailling) zeros.
    dms = [x.lstrip('0') for x in dms]
    # There may be empty fields after zero-stripping, replace them with zero.
    dms = ['0' if x == '' else x for x in dms]
    # Convert to float:
    dms = [float(x) for x in dms]
    # Convert to decimal:
    dec = (dms[0]
           + (dms[1] / 60.0)
           + dms[2] / 3600.0)
    return dec
