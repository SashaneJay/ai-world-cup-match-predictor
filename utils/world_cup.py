WORLD_CUP_2026_TEAMS = [
    "Algeria",
    "Argentina",
    "Australia",
    "Austria",
    "Belgium",
    "Bosnia and Herzegovina",
    "Brazil",
    "Canada",
    "Cape Verde",
    "Colombia",
    "Croatia",
    "Curaçao",
    "Czechia",
    "DR Congo",
    "Ecuador",
    "Egypt",
    "England",
    "France",
    "Germany",
    "Ghana",
    "Haiti",
    "Iran",
    "Iraq",
    "Ivory Coast",
    "Japan",
    "Jordan",
    "Mexico",
    "Morocco",
    "Netherlands",
    "New Zealand",
    "Norway",
    "Panama",
    "Paraguay",
    "Portugal",
    "Qatar",
    "Saudi Arabia",
    "Scotland",
    "Senegal",
    "South Africa",
    "South Korea",
    "Spain",
    "Sweden",
    "Switzerland",
    "Tunisia",
    "Turkey",
    "United States",
    "Uruguay",
    "Uzbekistan",
]


TEAM_ALIASES = {
    "Czechia": "Czech Republic",
}


def get_dataset_team_name(display_name):
    return TEAM_ALIASES.get(
        display_name,
        display_name
    )