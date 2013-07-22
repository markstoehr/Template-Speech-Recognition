#
#
#  Original from Gustav Larsson
#  https://github.com/gustavla/vision-research/detector/settings.py
#


import ConfigParser, re, os

def load_settings(fp):
    conf = ConfigParser.ConfigParser(dict_type=dict)
    conf.readfp(fp)
    d = {}
    for section in conf.sections():
        d[section] = {}
        for k, v in conf.items(section):
            match = re.match(r"^\s*\[\s*([A-Za-z0-9_]+)\s*\]\s*$", v.split('#')[0])
            try:
                if match is not None:
                    d[section][k] = d[match.group(1)]
                else:
                    d[section][k] = eval(v)
            except:
                import pdb; pdb.set_trace()
    return d

def argparse_settings(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('settings', metavar='<settings file>', type=argparse.FileType('r'), help='Filename of settings file')

    args = parser.parse_args()
    settings_file = args.settings
    return load_settings(settings_file)
