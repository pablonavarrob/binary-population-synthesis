import scaling_relations
import galah
import directories
import os
import sys
import socket
import numpy as np
import pandas as pd
computer_name = socket.gethostname()

# ---- Control Panel ----------------------------------------------------------- Control Panel
# Initialize directories
dirs = directories.directories()

# ------------------------------------------------------------------------------
def get_mass_primary_EggletonBook():
    """ Obtains the primary mass from a Salpter IMF and a random uniform number."""
    mass_primary = 0
    while mass_primary <= 0.65:
        X_1 = np.random.uniform(0,1)
        mass_primary = 0.3 * (X_1 / ( 1 - X_1))**0.55

    return mass_primary

def get_mass_ratio_EggletonBook(mass_primary):
    """ It begins from the period distribution, which we don't directly use. """

    # Define the period distribution
    X_2 = np.random.uniform(0,1)
    alpha_prime = 0.1 * mass_primary**1.5
    alpha = (3.5 + 1.3*alpha_prime)/(1 + alpha_prime)
    P = (5e4/mass_primary**2)*(X_2/(1-X_2))**alpha

    # Now define the inverse mass-ratio distribution
    X_3 = np.random.uniform(0,1)
    beta_prime = (0.1 * P**0.5)*(mass_primary + 0.5)
    beta = (2.5 + 0.7*beta_prime)/(1+beta_prime)
    q = 1 - X_3**beta

    return q

def generate_binarySystems(numberOfBinaries, stars_run):

    binaryStars = []
    ids_used_stars = []
    teffRange_primary = 75  # Plus / minus Kelvin value for the crossmatching
    fe_hRange_secondary = 0.025 # dex
    teffRange_secondary = 75 # plus/minus Kelvin

    while len(binaryStars) < numberOfBinaries:
        # --- To be done in every iteration ------------------------------------

        # The goal of this test is to check if we can recover a realistic mass
        # ratio from basically an IMF primary and a randomly chosen secondary

        stellarMass_primary = get_mass_primary_EggletonBook()
        # massRatio = get_mass_ratio_EggletonBook(stellarMass_primary)
        # stellarMass_secondary = massRatio * stellarMass_primary

        # Use the new scaling relations:
        temperature_PrimaryStar = scaling_relations.classic_MTR(
            stellarMass_primary)
        # temperature_SecondaryStar = scaling_relations.classic_MTR(
        #     stellarMass_secondary)

        # luminosityRatio = scaling_relations.classic_MLR(stellarMass_secondary) / \
        #     scaling_relations.classic_MLR(stellarMass_primary)

        # Main loop
        try: # Tries to find primary

            # Define the mask to obtain the GALAH stars
            mask_candidatesPrimary = (abs(temperature_PrimaryStar - stars_run['teff']) <= teffRange_primary)

            # Find a candidate the matches the defined conditions
            primaryStar_Candidate = np.random.permutation(stars_run[mask_candidatesPrimary])[0]

            if primaryStar_Candidate['sobject_id'] in ids_used_stars:
                print('Primary star skipped, was used and on the list already.')
                pass

            try: # Tries to find secondary
                # Secondary star should be of almost the same metallicity, have lower temperature than the
                # primary star and higher logg
                mask_candidatesSecondary =  ((primaryStar_Candidate['teff'] > stars_run['teff']) &
                                            (primaryStar_Candidate['logg'] < stars_run['logg']) &
                                            (abs(primaryStar_Candidate['fe_h'] - stars_run['fe_h']) <= fe_hRange_secondary))

                # The permutation is to pick the a random candidate each time
                secondaryStar_Candidate = np.random.permutation(stars_run[mask_candidatesSecondary])[0]

                # Check if the secondary has already been used
                if secondaryStar_Candidate['sobject_id'] in ids_used_stars:
                    print('Secondary star skipped, was used and on the list already. <--')
                    pass

                # Compute the derived quantities
                stellarMass_secondary = (secondaryStar_Candidate['teff']/5778)**(1/0.38) # Basically the inverse MTR
                luminosityRatio = scaling_relations.classic_MLR(stellarMass_secondary) / \
                    scaling_relations.classic_MLR(stellarMass_primary)
                massRatio = stellarMass_secondary/stellarMass_primary

                binarySystem_data = {'mass_A': stellarMass_primary,
                                     'teff_A': primaryStar_Candidate['teff'],
                                     'logg_A': primaryStar_Candidate['logg'],
                                     'feh_A': primaryStar_Candidate['fe_h'],
                                     'mass_B': stellarMass_secondary,
                                     'teff_B': secondaryStar_Candidate['teff'],
                                     'logg_B': secondaryStar_Candidate['logg'],
                                     'feh_B': secondaryStar_Candidate['fe_h'],
                                     'mass ratio': massRatio,
                                     'lum ratio': luminosityRatio,
                                     'comp_teff_A': temperature_PrimaryStar,
                                     'id_A': primaryStar_Candidate['sobject_id'],
                                     'id_B': secondaryStar_Candidate['sobject_id']}

                ids_used_stars.append(primaryStar_Candidate['sobject_id'])
                ids_used_stars.append(secondaryStar_Candidate['sobject_id'])
                binaryStars.append(binarySystem_data)
                print(len(binaryStars))

            except IndexError:
                # When no secondary star with the given parameters could be found.
                pass

        except IndexError:
            # Index error happens when no primary stars has been found.
            pass

    binaryStars = pd.DataFrame(binaryStars)

    return binaryStars

if __name__ == "__main__":
    # ---- Import/Filter Data ---------------------------
    # Gets the data from GALAH for the stellar synthesis
    galah_data = galah.GALAH_survey()
    galah_data.get_stars_run()

    # ---- Run the population simulation ----------------
    numberOfBinaries = 5000
    binaryStars_generated = generate_binarySystems(
        numberOfBinaries, galah_data.stars_run)

    binaryStars_generated.to_csv(dirs.data +
     ('BinaryPopulation_{}_onlyM_primary_is_random.csv').format(numberOfBinaries), index=False)
