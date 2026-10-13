"""Curated common first names (English and Spanish) used by the name guard.

Hand-picked, not exhaustive: the guard only needs names that also occur as surnames of
tracked players or as first-name aliases, and the tracked roster's own first names are
added at build time (see ``fansent.names``). Extend freely; entries are lowercase,
accent-folded.
"""

_RAW = """
    aaron abraham adam adrian alan albert alex alexander alfredo andres andrew angel anthony
    antonio arthur austin ben benjamin bobby brad bradley brandon brian bruce bryan caleb carlos
    carter charles chris christian christopher clayton cody cole colin connor corey cristian
    dale dan daniel danny darren david dean dennis derek diego dominic donald douglas drew dustin
    dylan eddie edgar eduardo edward edwin eli elias emmanuel eric erik ernesto ethan eugene evan
    felix fernando francisco frank franklin fred gabriel garrett gary gavin george gerald gerardo
    gordon grant greg gregory gustavo harold harry henry hunter ian isaac isaiah ivan jack jackson
    jacob jaime jake james jared jason javier jay jeff jeffrey jeremy jerry jesse jesus jim jimmy
    joe joel john johnny jonathan jordan jorge jose joseph josh joshua juan julian justin keith
    kevin kyle lance larry lawrence leo leonardo logan lucas luis luke manuel marcus mario mark
    martin mason matt matthew max michael miguel mike mitchell nathan neil nelson nicholas nick
    noah oliver omar oscar owen pablo patrick paul pedro peter philip phillip rafael ralph ramon
    randy raul ray raymond reid ricardo richard rick riley robert roberto rodrigo roger roman ronald ross
    roy russell ryan sam samuel santiago scott sean sebastian sergio seth shane shawn simon spencer
    stephen steve steven taylor terry thomas tim timothy todd tom tommy travis trevor tristan
    tyler victor vincent wade walker wallace walter warren wayne wesley will william willie
    xavier zach zachary
    """

COMMON_FIRST_NAMES: frozenset[str] = frozenset(_RAW.split())
