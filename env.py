"""A neural network training environment for texas-holdem poker"""

__author__ = "Felix Sondhauss"
__credits__ = ["Felix Sondhauss"]
__version__ = "1.0.0"

from copy import copy
import numpy as np


STARTING_MONEY = 200
MONEY_MAX = STARTING_MONEY * 5

MIN_BET = 5


class Card:
    face = 0
    value = 0

    def __init__(self, face, value):
        self.face = face
        self.value = value

    def flatten(self) -> int:
        # turn a 2d vector into a scalar (0 = no card)
        return max((self.face + 4 * self.value) - 3, 0) / 52

    def __lt__(self, other):
        return self.value < other.value

    def __str__(self) -> str:
        return f"({self.getValueName(self.value)} {self.getFaceName(self.face)})"

    def __repr__(self) -> str:
        return self.__str__()

    def getFaceName(self, face) -> str:
        match face:
            case 0:
                return "♥"
            case 1:
                return "♦"
            case 2:
                return "♠"
            case 3:
                return "♣"

    def getValueName(self, value) -> str:
        match value:
            case 1:
                return "A"
            case 11:
                return "J"
            case 12:
                return "Q"
            case 13:
                return "K"
            case _:
                return value


class Deck:
    cards = []

    def __init__(self) -> None:
        self.cards = [Card(i, j) for i in range(4) for j in range(1, 14)]
        np.random.shuffle(self.cards)

    def takeCard(self) -> Card:
        return self.cards.pop()


class Player:
    money = 0
    bet = 0
    out = False
    inRound = True

    def __init__(self, deck):
        self.card1 = deck.takeCard()
        self.card2 = deck.takeCard()
        self.money = STARTING_MONEY
        self.bet = 0

    def __str__(self) -> str:
        out = " out" if self.out else ""
        return f"(money: {self.money}, bet:{self.bet} cards:{self.card1}, {self.card2}{out})"

    def __repr__(self) -> str:
        return self.__str__()

    def action(self, choice):
        return choice


class PokerEnv:
    MONEY_REWARD = 1
    PLAYER_N = 5
    OBSERVATION_SPACE_VALUES = (PLAYER_N + PLAYER_N + PLAYER_N + 7,)
    ACTION_SPACE_SIZE = 4
    GAMES_N = 20
    startingPlayer = 0

    def reset(self):
        self.games = 0

        self.deck = Deck()
        self.cards = [self.deck.takeCard() for i in range(3)]
        self.players = [Player(self.deck) for i in range(self.PLAYER_N)]
        self.round = 0
        self.startingPlayer = (self.startingPlayer + 1) % self.PLAYER_N
        self.currentPlayer = self.startingPlayer
        self.playersInGame = copy(self.players)
        self.previousBet = MIN_BET

        self.episode_step = 0

        observations = []

        for j, player in enumerate(self.players):
            playersInRound = [
                float(self.players[(i + j) % self.PLAYER_N].inRound)
                for i in range(self.PLAYER_N)
            ]
            playerMoney = [
                self.players[(i + j) % self.PLAYER_N].money / MONEY_MAX
                for i in range(self.PLAYER_N)
            ]
            playerBet = [
                self.players[(i + j) % self.PLAYER_N].bet / MONEY_MAX
                for i in range(self.PLAYER_N)
            ]
            outCards = [card.flatten() for card in self.pad(self.cards)]

            observations.append(
                playersInRound
                + playerMoney
                + playerBet
                + [
                    self.players[j].card1.flatten(),
                    self.players[j].card2.flatten(),
                ]
                + outCards
            )

        return (
            observations,
            self.currentPlayer,
        )

    def pad(self, input) -> list:
        # guarantees the constant length of the amount of cards
        return input + [Card(0, 0)] * (5 - len(input))

    def step(self, agent_action, agent_n):
        self.episode_step += 1

        # agent action
        match agent_action:
            # fold
            case 0:
                self.playersInGame.remove(self.players[agent_n])
                self.players[agent_n].inRound = False

            # call
            case 1:
                self.players[agent_n].money -= (
                    self.previousBet - self.players[agent_n].bet
                )
                self.players[agent_n].bet = self.previousBet

                if self.players[agent_n].money < 0:
                    self.players[agent_n].bet += self.players[agent_n].money
                    self.players[agent_n].money = 0

            # raise
            case 2:
                self.players[agent_n].money -= (
                    self.previousBet - self.players[agent_n].bet
                )
                self.players[agent_n].bet = self.previousBet

                self.players[agent_n].money -= self.players[agent_n].bet

                self.players[agent_n].bet *= 2

                if self.players[agent_n].money < 0:
                    self.players[agent_n].bet += self.players[agent_n].money
                    self.players[agent_n].money = 0

            # all in
            case 3:
                self.players[agent_n].bet += self.players[agent_n].money
                self.players[agent_n].money = 0

        playerMoney = [player.money for player in self.playersInGame]
        playerBet = [player.bet for player in self.playersInGame]

        done = False

        if (
            agent_n == (self.startingPlayer) % self.PLAYER_N
            and (min(playerBet) == max(playerBet) or sum(playerMoney) == 0)
        ) or len(self.playersInGame) == 1:
            # advance to the next game
            if self.round == 1 or len(self.playersInGame) == 1:
                self.cards.append(self.deck.takeCard())

                # add money to winner
                if len(self.playersInGame) == 1:
                    self.playersInGame[0].money += sum(playerBet)
                else:
                    winner = self.checkHands(self.cards, self.playersInGame)
                    if winner != -1:
                        for i, player in enumerate(self.playersInGame):
                            if winner[i]:
                                player.money += sum(playerBet) // winner.count(True)

                # reset game information
                self.deck = Deck()
                self.cards = [self.deck.takeCard() for i in range(3)]
                self.playersInGame = []
                self.previousBet = MIN_BET
                for player in self.players:
                    # reset money
                    if player.money < MIN_BET:
                        player.out = True
                        player.inRound = False
                    player.bet = 0
                    # re-add those players that are not out
                    if not player.out:
                        player.inRound = True
                        self.playersInGame.append(player)

                inGame = 0
                for player in self.players:
                    if not player.out:
                        inGame += 1
                if self.games == self.GAMES_N or inGame <= 1:
                    done = True
                self.games += 1
                self.round = 0

                if self.players[self.startingPlayer].out:
                    for i in range(self.PLAYER_N):
                        self.startingPlayer = (self.startingPlayer + 1) % self.PLAYER_N
                        if not self.players[self.startingPlayer].out:
                            break

            else:
                self.round += 1
                self.cards.append(self.deck.takeCard())
        # Next player
        for i in range(self.PLAYER_N):
            self.currentPlayer = (self.currentPlayer + 1) % self.PLAYER_N
            if self.players[self.currentPlayer].inRound:
                break

        # What bet to call to
        if self.playersInGame.count(self.players[agent_n]) >= 1:
            self.previousBet = max(self.players[agent_n].bet, MIN_BET)

        playersInRound = [
            float(self.players[(i + self.currentPlayer) % self.PLAYER_N].inRound)
            for i in range(self.PLAYER_N)
        ]
        playerMoney = [
            self.players[(i + self.currentPlayer) % self.PLAYER_N].money / MONEY_MAX
            for i in range(self.PLAYER_N)
        ]
        playerBet = [
            self.players[(i + self.currentPlayer) % self.PLAYER_N].bet / MONEY_MAX
            for i in range(self.PLAYER_N)
        ]
        outCards = [card.flatten() for card in self.pad(self.cards)]

        return (
            playersInRound
            + playerMoney
            + playerBet
            + [
                self.players[self.currentPlayer].card1.flatten(),
                self.players[self.currentPlayer].card2.flatten(),
            ]
            + outCards,
            self.currentPlayer,
            self.players[self.currentPlayer].money,
            done,
        )

    def getRewards(self):
        return [player.money for player in self.players]

    def render(self):
        print(self.cards)
        for player in self.players:
            print(player)

    def checkHands(self, mainCards, players) -> list:
        scores = []
        for player in players:

            cards = copy(mainCards)
            cards.append(player.card1)
            cards.append(player.card2)

            scores.append(self.getScore(cards))

        if max(scores) <= self.getScore(mainCards):
            return -1
        o = []

        # create an array of true and false so that ties can happen
        for score in scores:
            if score == max(scores):
                o.append(True)
                continue
            o.append(False)
        return o

    def hasCardValue(self, cards, value) -> bool:
        for card in cards:
            # Ace can be 1 and 14
            if card.value % 13 == value % 13:
                return True
        return False

    def hasCardFaceValue(self, cards, face, value) -> bool:
        for card in cards:
            # Ace can be 1 and 14
            if card.value % 13 == value % 13 and face == card.face:
                return True
        return False

    def hasCardsOfValue(self, cards, value) -> int:
        amount = 0
        for card in cards:
            if card.value == value:
                amount += 1

        return amount

    def hasCardsOfAFace(self, cards, face) -> int:
        amount = 0
        for card in cards:
            if card.face == face:
                amount += 1

        return amount

    # TODO: optimize (Important)
    def getScore(self, cards: list) -> int:
        cards.sort(reverse=True)

        for card in cards:
            # Straight flush & Royal flush
            if (
                self.hasCardFaceValue(cards, card.face, card.value - 1)
                and self.hasCardFaceValue(cards, card.face, card.value - 2)
                and self.hasCardFaceValue(cards, card.face, card.value - 3)
                and self.hasCardFaceValue(cards, card.face, card.value - 4)
            ):
                return card.value + 13 * 8

            for card in cards:
                # Four of a kind
                if self.hasCardsOfValue(cards, card.value) == 4:
                    return card.value + 13 * 7

            for card in cards:
                # Full house
                if self.hasCardsOfValue(cards, card.value) == 3:
                    for card2 in cards:
                        if (
                            self.hasCardsOfValue(cards, card2.value) >= 2
                            and card.value != card2.value
                        ):
                            return max(card.value, card2.value) + 13 * 6

            for card in cards:
                # Flush
                if self.hasCardsOfAFace(cards, card.face) >= 5:
                    return card.value + 13 * 5

            for card in cards:
                # Straight
                if (
                    self.hasCardValue(cards, card.value - 1)
                    and self.hasCardValue(cards, card.value - 2)
                    and self.hasCardValue(cards, card.value - 3)
                ):
                    return card.value + 13 * 4

            for card in cards:
                # Three of a kind
                if self.hasCardsOfValue(cards, card.value) == 3:
                    return card.value + 13 * 3

            for card in cards:
                # Pair
                if self.hasCardsOfValue(cards, card.value) == 2:

                    # Two Pairs
                    for card2 in cards:
                        if (
                            self.hasCardsOfValue(cards, card2.value) == 2
                            and card.value != card2.value
                        ):
                            return card.value + 13 * 2

                    return card.value + 13

        # Highcard
        return cards[0].value
