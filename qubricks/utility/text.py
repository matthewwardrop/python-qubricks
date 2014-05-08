import sys
COLOURS = (
    'BLACK', 'RED', 'GREEN', 'YELLOW',
    'BLUE', 'MAGENTA', 'CYAN', 'WHITE'
)

def colour_text(text, colour_name='WHITE', bold=False):
    if colour_name in COLOURS:
        return '\033[{0};{1}m{2}\033[0m'.format(
            int(bold), COLOURS.index(colour_name) + 30, text)
    sys.stderr.write('ERROR: "{0}" is not a valid colour.\n'.format(colour_name))
    sys.stderr.write('VALID COLOURS: {0}.\n'.format(', '.join(COLOURS)))
