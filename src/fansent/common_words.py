"""Curated ordinary English words that also occur as player surnames.

Hand-picked, not derived from a dictionary: a dictionary flags nearly every surname
(``martin``, ``webb``, ``bell``), whereas the goal is words that show up in lowercase
running text where a surname match would be a false positive ('a little rough',
'early innings', 'a good story'). Entries are everyday nouns, adjectives, verbs and
adverbs, plus English occupational/nature/colour surnames from MLB rosters past and present.
Names that are primarily given names (``martin``, ``hunter``) stay in
``fansent.first_names``; the two guards are independent. Only words of 4+ letters matter,
since shorter surnames are never matched bare. Extend freely; entries are lowercase,
accent-folded. Used by ``fansent.names`` to require a capitalized surname token.
"""

_RAW = """
    able acre alley angel arrow baker ball band bank banks bare barber bark barn barrel base
    bass bates battle beach beam bean bear beard beat beck bell belt bench best bird bishop bits
    black blade bland blank blind block blood blow blue board boat bolt bond bone book boot boss
    bound bowl bowman box brace brake brave bread break brick bridge bright brook brown buck
    bull burn burns bush butter cabin cage cain cake camp candy cane card care carpenter carry
    cart case cash cast cave chance chandler charm chase cheek church clay clean clear clerk
    cliff close cloud clue coach coast cook cool copy cord core cork corn count court cove craft
    crane crash cream creek crew cross crow crown crush cup cut dance dark dash dawn day deal
    deer dent dirt dive dock doll door down drake draper dream drew drive drop duck duke dust
    eagle early earl east edge fair fall fan farm fast fee fell fence field fields fine finch
    fire fish fisher flag flat fleet flood floor flower fold food fool foot force ford forest
    fork fort fox frame free fresh frost game gate gift gill glass glove goal gold good grace
    grade grain grand grant grass gray green grey grove guard guide hair hall hand hands hard
    harm harper hart harvest hawk head heart heat heath heaven hedge held hell helm herd hero
    hill hoke hold hole home hood hook hope horn horse host house hunt hurt ice iron jack jam
    jewel joy judge jump just keen keep key kick kind king kiss knight lace lack lake lamb land
    lane large last late law lawn lead leaf lean left lend less light lime line link lion list
    little load lock lodge long look lord love low luck lunch mace main major mark marsh mask
    mass match may meal meat melon mild mill mind miner mint mirror miss mist mix moon moor moss
    mount mouse nail name near neat neck needle nest net new news nice night noble noon north
    nose note oak oats ocean odd oil old open orange pace pack page paint pair palm park past
    path pay peace peach peak pearl pen pepper pick pike pilot pine pitch place plain plant
    plate play plum point poor port post pound power price pride prime proof pump punch pure
    quick quiet race rain ranger rank rare rash ray reach read ready reason reed rest rice rich
    ride right ring rise river rivers road robin rock rod roof room root rope rose round row
    royal ruby rule run rush sage sail salt sand sands save scale scarlet school score scout sea
    seal seat see seed sense shade sheets shell shield shift shine ship shop shore short shot
    shoulder show side sign silver sink skill skin sky slate sleep slide slow small smart smith
    smoke snow soft soil sound south sparks spear speed spell spin spirit spot spring square
    staff stage stair stake stamp stand star start state stay steady steel stem step stern stick
    still stock stone stop store storm story straight strange stream street strike string strong
    style sugar summer sun sunday sure swan sweet swift swing sword table tail talk tall tank
    tape taste team tell temple thin thorn thunder tide tight time tin tip toast token tone tool
    tooth top touch tower town toy trace track trade trail train tree trial trick trout true
    trust turn twin uncle union upper vale valley vault view voice wade wage wagon wait wake
    walk wall want ward warm wash watch water wave way wealth weather well west wheat wheel
    white whole wild will wind wine wing winter wise wolf wood woods word work world worth yard
    year young
    """

COMMON_WORDS: frozenset[str] = frozenset(_RAW.split())
