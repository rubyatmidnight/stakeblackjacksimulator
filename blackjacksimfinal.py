# Blackjack sim
import secrets
import hmac
import hashlib
import numpy as np
from typing import Generator, List, Tuple, Dict
from dataclasses import dataclass
import concurrent.futures
import time

# Card configs
CARDS = [
    '♦2', '♥2', '♠2', '♣2', '♦3', '♥3', '♠3', '♣3', '♦4', '♥4',
    '♠4', '♣4', '♦5', '♥5', '♠5', '♣5', '♦6', '♥6', '♠6', '♣6',
    '♦7', '♥7', '♠7', '♣7', '♦8', '♥8', '♠8', '♣8', '♦9', '♥9',
    '♠9', '♣9', '♦10', '♥10', '♠10', '♣10', '♦J', '♥J', '♠J',
    '♣J', '♦Q', '♥Q', '♠Q', '♣Q', '♦K', '♥K', '♠K', '♣K', '♦A',
    '♥A', '♠A', '♣A'
]

CARD_VALUES = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    '10': 10, 'J': 10, 'Q': 10, 'K': 10, 'A': 11
}

@dataclass
class Hand:
    cards: List[str]
    total: int
    soft: bool

@dataclass
class GameResult:
    betSize: float
    outcome: float  # 1.5 for natural, 1 for win, 0 for push, -1 for loss
    playerCards: List[str]
    dealerCards: List[str]
    playerTotal: int
    dealerTotal: int

# RNG funcs
def genSeed(length: int = 64) -> str:
    return ''.join(secrets.choice("abcdef0123456789") for _ in range(length))

def byteGen(serverSeed: str, clientSeed: str, nonce: str, cursor: int) -> Generator[int, None, None]:
    currRound = cursor // 32
    currCursor = cursor - (currRound * 32)
    
    while True:
        msg = f"{clientSeed}:{nonce}:{currRound}"
        hmacObj = hmac.new(serverSeed.encode(), msg.encode(), hashlib.sha256)
        buff = hmacObj.digest()
        
        while currCursor < 32:
            yield buff[currCursor]
            currCursor += 1
        currCursor = 0
        currRound += 1

def genFloats(serverSeed: str, clientSeed: str, nonce: str, cursor: int, count: int) -> List[float]:
    rng = byteGen(serverSeed, clientSeed, nonce, cursor)
    bytes = []
    
    while len(bytes) < count * 4:
        bytes.append(next(rng))
    
    return [sum(val / (256 ** (i + 1)) for i, val in enumerate(bytes[i:i+4])) 
            for i in range(0, len(bytes), 4)]

# Game funcs
def calcHand(cards: List[str]) -> Hand:
    total = 0
    aces = 0
    
    for card in cards:
        val = card[1:]
        if val != 'A':
            total += CARD_VALUES[val]
        else:
            aces += 1
    
    soft = False
    for _ in range(aces):
        if total <= 10:
            total += 11
            soft = True
        else:
            total += 1
    
    if total > 21 and soft:
        total -= 10
        soft = False
    
    return Hand(cards, total, soft)

def verifyCards(serverSeed: str, clientSeed: str, nonce: str, count: int = 52) -> List[Tuple[float, int, str]]:
    floats = genFloats(serverSeed, clientSeed, nonce, 0, count)
    return [(f, int(f * 52), CARDS[int(f * 52)]) for f in floats]

def getHands(serverSeed: str, clientSeed: str, nonce: str) -> Dict[str, Hand]:
    cards = verifyCards(serverSeed, clientSeed, nonce, 4)
    return {
        'player': calcHand([c[2] for c in cards[:2]]),
        'dealer': calcHand([c[2] for c in cards[2:4]])
    }

class BlackjackSim:
    def __init__(self, serverSeed: str, clientSeed: str):
        self.serverSeed = serverSeed
        self.clientSeed = clientSeed
        self.nonce = 1
    
    def _shouldDouble(self, hand: Hand, dUpcard: str) -> bool:
        if not hand.soft and hand.total == 11:
            return True
        if not hand.soft and hand.total == 10 and dUpcard not in ['10', 'A']:
            return True
        if not hand.soft and hand.total == 9 and dUpcard in ['3', '4', '5', '6']:
            return True
        return False

    def _shouldHit(self, hand: Hand, dUpcard: str) -> bool:
        if hand.total >= 17:
            return False
            
        if hand.soft and hand.total <= 17:
            return True
            
        if hand.total <= 11:
            return True
        elif hand.total == 12:
            return dUpcard not in ['4', '5', '6']
        elif hand.total <= 16:
            return dUpcard not in ['2', '3', '4', '5', '6']
            
        return False
    
    def _shouldSplit(self, cardVal: str, dUpcard: str) -> bool:
        if cardVal in ['A', '8']:
            return True
        if cardVal in ['2', '3', '7'] and dUpcard in ['2', '3', '4', '5', '6', '7']:
            return True
        if cardVal == '6' and dUpcard in ['2', '3', '4', '5', '6']:
            return True
        if cardVal == '4' and dUpcard in ['5', '6']:
            return True
        if cardVal == '9' and dUpcard in ['2', '3', '4', '5', '6', '8', '9']:
            return True
        return False

    def _playDealer(self, cards: List[Tuple], startIdx: int) -> List[str]:
        dCards = [cards[0][2], cards[1][2]]
        dHand = calcHand(dCards)
        idx = startIdx
        
        while dHand.total < 17:  # Stand on S17
            if idx >= len(cards):
                break
            dCards.append(cards[idx][2])
            dHand = calcHand(dCards)
            idx += 1
            
        return dCards

    def simGame(self, clientSeed: str, nonce: int) -> GameResult:
        hands = getHands(self.serverSeed, clientSeed, str(nonce))
        pHand = hands['player']
        dUpcard = hands['dealer'].cards[0]
        cards = verifyCards(self.serverSeed, clientSeed, str(nonce), 52)
        dHand = calcHand([cards[2][2], cards[3][2]])
        
        # Check naturals
        pNatural = len(pHand.cards) == 2 and pHand.total == 21
        dNatural = dHand.total == 21
        
        # Handle naturals
        if pNatural or dNatural:
            if pNatural and dNatural:
                return GameResult(1, 0, pHand.cards, dHand.cards, 21, 21)
            elif pNatural:
                return GameResult(1, 1.5, pHand.cards, dHand.cards, 21, dHand.total)
            else:
                return GameResult(1, -1, pHand.cards, dHand.cards, pHand.total, 21)
        
        # Split check
        pCards = pHand.cards.copy()
        idx = 4
        betMult = 1.0
        
        if (len(pCards) == 2 and pCards[0][1:] == pCards[1][1:] and 
            self._shouldSplit(pCards[0][1:], dUpcard[1:])):
            # Handle split
            card1, card2 = pCards
            if card1[1:] == 'A':  # Split aces
                return GameResult(2, 1 if calcHand([card1, cards[idx][2]]).total == 21 
                                or calcHand([card2, cards[idx+1][2]]).total == 21 else -1,
                                [card1, card2], dHand.cards, 21 if calcHand([card1, cards[idx][2]]).total == 21 
                                else calcHand([card2, cards[idx+1][2]]).total, dHand.total)
            pCards = [card1]
            pHand = calcHand(pCards)
        
        # Check double
        if len(pCards) == 2 and self._shouldDouble(pHand, dUpcard[1:]):
            betMult = 2.0
            pCards.append(cards[idx][2])
            pHand = calcHand(pCards)
            idx += 1
        else:
            # Normal play
            while self._shouldHit(pHand, dUpcard[1:]):
                pCards.append(cards[idx][2])
                pHand = calcHand(pCards)
                idx += 1
                
                if pHand.total > 21:
                    return GameResult(betMult, -1, pCards, [dUpcard], pHand.total, dHand.total)
        
        # Dealer plays
        dealerCards = self._playDealer(cards[2:], idx)
        dealerHand = calcHand(dealerCards)
        
        # Compare hands
        if dealerHand.total > 21:
            return GameResult(betMult, 1, pCards, dealerCards, pHand.total, dealerHand.total)
        elif pHand.total > dealerHand.total:
            return GameResult(betMult, 1, pCards, dealerCards, pHand.total, dealerHand.total)
        elif pHand.total < dealerHand.total:
            return GameResult(betMult, -1, pCards, dealerCards, pHand.total, dealerHand.total)
        else:
            return GameResult(betMult, 0, pCards, dealerCards, pHand.total, dealerHand.total)

    def simGames(self, numGames: int, threads: int = 8) -> Dict:
        results = []
        winUnits = 0
        losses = pushes = 0
        totalReturn = 0
        startTime = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as exec:
            futures = [exec.submit(self.simGame, f"{self.clientSeed}_{i}", 
                      self.nonce + i) for i in range(numGames)]
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                
                if result.outcome > 0:
                    winUnits += result.outcome * result.betSize
                    totalReturn += result.outcome * result.betSize
                elif result.outcome < 0:
                    losses += 1
                    totalReturn += result.outcome * result.betSize
                else:
                    pushes += 1
                
                results.append(result)
        
        totalHands = winUnits + losses + pushes
        
        return {
            'numGames': numGames,
            'winUnits': winUnits,
            'losses': losses,
            'pushes': pushes,
            'winRate': (winUnits / totalHands) * 100,
            'lossRate': (losses / totalHands) * 100,
            'pushRate': (pushes / totalHands) * 100,
            'houseEdge': (-totalReturn / numGames) * 100,
            'variance': np.var([r.outcome for r in results]),
            'timeElapsed': time.time() - startTime,
            'avgReturn': totalReturn / numGames
        }

def verify(serverSeed: str, clientSeed: str, nonce: str):
    results = verifyCards(serverSeed, clientSeed, nonce, 52)
    hands = getHands(serverSeed, clientSeed, nonce)
    
    print(f"\n{'='*50}")
    print("Verification Results:")
    print(f"Server Seed: {serverSeed}")
    print(f"Client Seed: {clientSeed}")
    print(f"Nonce: {nonce}")
    print(f"{'='*50}\n")
    
    print("Initial Hands:")
    print(f"Player: {' '.join(hands['player'].cards)} (Total: {hands['player'].total}{'s' if hands['player'].soft else ''})")
    print(f"Dealer: {' '.join(hands['dealer'].cards)} (Total: {hands['dealer'].total}{'s' if hands['dealer'].soft else ''})")
    print(f"\n{'='*50}\n")
    
    for i, (f, idx, card) in enumerate(results, 1):
        print(f"Card {i:2d}: {card:4s} (Index: {idx:2d}, Float: {f:.8f})")

def runSim(serverSeed: str = None, clientSeed: str = None, numGames: int = 100000):
    if not serverSeed:
        serverSeed = genSeed(64)
    if not clientSeed:
        clientSeed = genSeed(12)
        
    sim = BlackjackSim(serverSeed, clientSeed)
    results = sim.simGames(numGames)
    
    print(f"\n{'='*50}")
    print(f"Simulation Results ({numGames:,} games):")
    print(f"Server Seed: {serverSeed}")
    print(f"Client Seed: {clientSeed}")
    print(f"Win Units: {results['winUnits']:.1f} ({results['winRate']:.2f}%)")
    print(f"Losses: {results['losses']:,} ({results['lossRate']:.2f}%)")
    print(f"Pushes: {results['pushes']:,} ({results['pushRate']:.2f}%)")
    print(f"House Edge: {results['houseEdge']:.2f}%")
    print(f"Average Return: {results['avgReturn']:.4f}")
    print(f"Variance: {results['variance']:.4f}")
    print(f"Time: {results['timeElapsed']:.2f}s")
    print(f"{'='*50}\n")
    
    return results

if __name__ == "__main__":
    # Example usage
    print("Running verification...")
    verify("test1", "test1", "1")
    
    print("\nRunning simulation...")
    # Custom seeds
    runSim(
        serverSeed="test1",
        clientSeed="test1",
        numGames=100000
    )
    
    # Random seeds
    print("\nRunning with random seeds...")
    runSim(numGames=100000)
