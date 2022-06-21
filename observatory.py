

def observ_param(obs):

    kpno = {
        'Name': "kpno",
        'Fullname': 'Kitt Peak National Observatory',
        'Longitude': '111:36.0',
        'Latitude': '31:57.8',
        'Altitude': 2120.,
        'Timezone': 7
    }

    ctio = {
        'Name': "ctio",
        'Fullname': "Cerro Tololo Interamerican Observatory",
        'Longitude': '70.815',
        'Latitude': '-30.16527778',
        'Altitude': 2215.,
        'Timezone': 4
    }

    eso = {
        'Name': "eso",
        'Fullname': "European Southern Observatory",
        'Longitude': '70:43.8',
        'Latitude': '-29:15.4',
        'Altitude': 2347,
        'Timezone': 4
    }

    lick = {
        'Name': "lick",
        'Fullname': "Lick Observatory",
        'Longitude': '121:38.2',
        'Latitude': '37:20.6',
        'Altitude': 1290,
        'Timezone': 8
    }

    mmto = {
        'Name': "mmto",
        'Fullname': "Multiple Mirror Telescope Observatory",
        'Longitude': '110:53.1',
        'Latitude': '31:41.3',
        'Altitude': 2600,
        'Timezone': 7
    }

    mmt = {
        'Name': "mmt",
        'Fullname': "Whipple Observatory",
        'Longitude': '110:53.1',
        'Latitude': '31.41.3',
        'Altitude': 2608,
        'Timezone': 7
    }

    flwo = {
        'Name': "flwo",
        'Fullname': "Whipple Observatory",
        'Longitude': '110:52:39',
        'Latitude': '31:40:51.4',
        'Altitude': 2320,
        'Timezone': 7
    }

    cfht = {
        'Name': "cfht",
        'Fullname': "Canada-France-Hawaii Telescope",
        'Longitude': '-155:28:10.48',
        'Latitude': '19:49:31.25',
        'Altitude': 4215,
        'Timezone': 10
    }

    lapalma = {
        'Name': "lapalma",
        'Fullname': "Roque de los Muchachos, La Palma",
        'Longitude': '17:52.8',
        'Latitude': '28:45.5',
        'Altitude': 2327,
        'Timezone': 0
    }

    mso = {
        'Name': "mso",
        'Fullname': "Mt. Stromlo Observatory",
        'Longitude': '210:58:32.4',
        'Latitude': '-35:19:14.34',
        'Altitude': 767,
        'Timezone': -10
    }

    sso = {
        'Name': "sso",
        'Fullname': "Siding Spring Observatory",
        'Longitude': '210:56:19.70',
        'Latitude': '-31:16:24.10',
        'Altitude': 1149,
        'Timezone': -10
    }

    aao = {
        'Name': "aao",
        'Fullname': "Anglo-Australian Observatory",
        'Longitude': '210:56:2.09',
        'Latitude': '-31:16:37.34',
        'Altitude': 1164,
        'Timezone': -10
    }

    mcdonald = {
        'Name': "mcdonald",
        'Fullname': "McDonald Observatory",
        'Longitude': '104.0216667',
        'Latitude': '30.6716667',
        'Altitude': 2075,
        'Timezone': 6
    }

    lco = {
        'Name': "lco",
        'Fullname': "Las Campanas Observatory",
        'Longitude': '70:42.1',
        'Latitude': '-29:0.2',
        'Altitude': 2282,
        'Timezone': 4
    }

    # Submitted by Alan Koski 1/13/93
    mtbigelow = {
        'Name': "mtbigelow",
        'Fullname': "Catalina Observatory: 61 inch telescope",
        'Longitude': '110:43.9',
        'Latitude': '32:25.0',
        'Altitude': 2510.,
        'Timezone': 7
    }

    # Revised by Daniel Durand 2/23/93
    dao = {
        'Name': "dao",
        'Fullname': "Dominion Astrophysical Observatory",
        'Longitude': '123:25.0',
        'Latitude': '48:31.3',
        'Altitude': 229,
        'Timezone': 8
    }

    # Submitted by Patrick Vielle 5/4/93
    spm = {
        'Name': "spm",
        'Fullname': "Observatorio Astronomico Nacional, San Pedro Martir",
        'Longitude': '115:29:13',
        'Latitude': '31:01:45',
        'Altitude': 2830,
        'Timezone': 7
    }

    # Submitted by Patrick Vielle 5/4/93
    tona = {
        'Name': "tona",
        'Fullname': "Observatorio Astronomico Nacional, Tonantzintla",
        'Longitude': '98:18:50',
        'Latitude': '19:01:58',
        'Timezone': 8
    }

    # Submitted by Don Hamilton 8/18/93
    palomar = {
        'Name': "Palomar",
        'Fullname': "The Hale Telescope",
        'Longitude': '116:51:46.80',
        'Latitude': '33:21:21.6',
        'Altitude': 1706.,
        'Timezone': 8
    }

    # Submitted by Pat Seitzer 10/31/93
    mdm = {
        'Name': "mdm",
        'Fullname': "Michigan-Dartmouth-MIT Observatory",
        'Longitude': '111:37.0',
        'Latitude': '31:57.0',
        'Altitude': 1938.5,
        'Timezone': 7
    }

    # Submitted by Ignacio Ferrin 9/1/94
    nov = {
        'Name': "NOV",
        'Fullname': "National Observatory of Venezuela",
        'Longitude': '70:52.0',
        'Latitude': '8:47.4',
        'Altitude': 3610,
        'Timezone': 4,
    }

    # Submitted by Alan Welty 10/28/94
    bmo = {
        'Name': "bmo",
        'Fullname': "Black Moshannon Observatory",
        'Longitude': '78:00.3',
        'Latitude': '40:55.3',
        'Altitude': 738,
        'Timezone': 5
    }

    # Submitted by Biwei JIANG 11/28/95
    bao = {
        'Name': "BAO",
        'Fullname': "Beijing XingLong Observatory",
        'Longitude': '242:25.5',
        'Latitude': '40:23.6',
        'Altitude': 950.,
        'Timezone': -8
    }

    # From Astronomical Almanac 1996
    keck = {
        'Name': "keck",
        'Fullname': "W. M. Keck Observatory",
        'Longitude': '155:28.7',
        'Latitude': '19:49.7',
        'Altitude': 4160,
        'Timezone': 10
    }

    # Padova Astronomical Observatory, Asiago, Italy.
    # Submitted by Lina Tomasella 6/11/96
    ekar = {
        'Name': "ekar",
        'Fullname': "Mt. Ekar 182 cm. Telescope",
        'Longitude': '348:25:07.92',
        'Latitude': '45:50:54.92',
        'Altitude': 1413.69,
        'Timezone': -1
    }

    # Submitted by Michael Ledlow 8/8/96
    apo = {
        'Name': "apo",
        'Fullname': "Apache Point Observatory",
        'Longitude': '105:49.2',
        'Latitude': '32:46.8',
        'Altitude': 2798.,
        'Timezone': 7
    }

    # Submitted by Michael Ledlow 8/8/96
    lowell = {
        'Name': "lowell",
        'Fullname': "Lowell Observatory",
        'Longitude': '111:32.1',
        'Latitude': '35:05.8',
        'Altitude': 2198.,
        'Timezone': 7
    }

    # Submitted by S.G. Bhargavi 8/12/96
    vbo = {
        'Name': "vbo",
        'Fullname': "Vainu Bappu Observatory",
        'Longitude': '281.1734',
        'Latitude': '12.57666',
        'Altitude': 725.,
        'Timezone': -5.5
    }

    omm = {
        'Name': "omm",
        'Fullname': "Observatoire du Mont-Mégantic",
        'Longitude': '-71:09:49',
        'Latitude': '45:26:57',
        'Altitude': 1111.,
        'Timezone': -5
    }

    try:
        exec('global observatoire; observatoire = {}'.format(obs))
        return observatoire
    except:
        print(obs+" n'est pas un observatoire valide. Choissez en un autre.")
