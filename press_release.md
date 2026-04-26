# New Magic The Gathering Win Probability Calculator

## Hook

This tool allows users to compare decklists within the standard metagame. Allowing for new players to quickly become engrained within the metagame and minimizing cost expenditure. 

## Problem Statement

Magic the Gathering as a game can be expensive to become involved in, both in terms of money and time. The game is notoriously complex, taking the time to learn interactions between cards and build a deck based on those can require more resources than available to the average person. If someone were to learn the metagame, they would be faced with an additional hurdle of cost. A competitve deck within standard will average around $400, requiring a significant amount of disposable income. The specific problem is identifying the outcome of a magic game given two decklists.

## Solution Description

This solution compares two of the most popular decks within the standard format: Dimir Excrutiator and Mono Green Landfall. Dimir Excrutiator is a combo deck that removes most of its opponents' deck from the game and then mill them out for the win. Mono Green Landfall leverages the games resource system to develop an advantage in material and win through combat damage. This solution utilizes principal component analysis to reduce dimensionality and then assigns a score to every card within the standard format. Each card is a document that is accessed within Mongo DB. Decklists are then imported with each card ranked by score. Lands tended to perform the best due to their flexibility.

## Chart

<img width="1485" height="1518" alt="image" src="https://github.com/user-attachments/assets/890a994f-e1d5-49fc-b0b5-666735bca3b2" />
