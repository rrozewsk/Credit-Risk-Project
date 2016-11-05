# November 05 Hack sesh
# =====================
#
# Today, we should make sure the following work:
#   + [ ] Create a Monte Carlo simulation for a discount curve Z(t),
#         using realistic perameters (by fitting with actual Libor 
#         data).
#   + [ ] For any company/rating, create a credit curve Q(t) given Z(t)
#         and at least one of the following methods:
#           - [ ] Bootstrapping from a bond yield curve (use the credit
#                 triangle). This requires inputting a recovery rate R.
#           - [ ] Bootstrapping from CDS data. This is mostly insensive
#                 to changes in R (so set R = 0.4 or something).
#            -[ ] Derive Q(t) from equity curve (using Merton's Model or
#                 equivalent).
#   + [ ] Make a script that allows the user to create a portfolio consisting
#         of CDS's and Bonds and apply exposure analysis.

