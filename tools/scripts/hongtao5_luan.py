#!/usr/bin/env python3
"""六安红桃5（对门一班）命令行版

规则要点（按用户给定规范实现核心逻辑）：
- 4人、2副牌共108张、固定队友：0<->2, 1<->3（Team0/Team1）。
- 发牌100张（每人25），留8张底牌。
- 亮主/反主：红桃5反主优先级最高，其次其他反主，再次普通亮主。
- 庄家=最终亮主者；庄家拿底牌后再扣8张。
- 主牌排序核心：红桃5 > 大王 > 小王 > 正参谋(主3) > 副参谋(同色3) > 主2 > 其他2 > 主A..4。
- 副牌：A>K>Q>J>10>9>8>7>6>5>4（副牌中无2/3，2/3都算主牌）。
- 牌型：单、对、三、四、拖拉机（连续对子）。
- 跟牌优先同花色同型；不能跟时可垫，可主毙（同型要求）。
- 计分：5=5分，10/K=10分；总分200。
- 最后一墩抠底：单x2，对x4，拖拉机x8（本实现按此）。
- 胜负：闲家(非庄家队) >=80 则闲家胜，否则庄家胜。

说明："甩牌"在实战中判定复杂，本版本未开放甩牌输入，仅实现核心可玩流程。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SUITS = ["♠", "♥", "♣", "♦"]
NORMAL_RANKS = ["4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
FULL_RANKS = ["2", "3", *NORMAL_RANKS]

# 固定队伍（对门一班）
TEAM_OF = {0: 0, 1: 1, 2: 0, 3: 1}
TEAM_PLAYERS = {0: (0, 2), 1: (1, 3)}


@dataclass(frozen=True)
class Card:
    rank: str
    suit: str = ""  # 王无花色
    joker: str = ""  # "SJ" / "BJ"

    def __str__(self) -> str:
        if self.joker:
            return "小王" if self.joker == "SJ" else "大王"
        return f"{self.suit}{self.rank}"


@dataclass
class BidAction:
    bidder: int
    trump_suit: str
    level: int  # 1 普通亮主，2 其他反主，3 红桃5反主


@dataclass
class Pattern:
    kind: str  # single,pair,triple,quad,tractor
    cards: List[Card]


class LuAnHongTao5:
    def __init__(self) -> None:
        self.hands: Dict[int, List[Card]] = {i: [] for i in range(4)}
        self.bottom: List[Card] = []

        self.trump_suit = "♥"
        self.dealer = 0
        self.dealer_team = 0
        self.attacker_team = 1

        self.scores = {0: 0, 1: 0}
        self.current_leader = 0
        self.last_trick_winner = 0
        self.last_trick_pattern = "single"

        self._deal_100_plus_8()
        self._bidding_phase()
        self._dealer_take_and_discard_bottom()
        self.current_leader = self.dealer

    # ---------- 牌堆与发牌 ----------
    def _build_deck(self) -> List[Card]:
        deck: List[Card] = []
        for _ in range(2):
            for s in SUITS:
                for r in FULL_RANKS:
                    deck.append(Card(rank=r, suit=s))
            deck.append(Card(rank="", joker="SJ"))
            deck.append(Card(rank="", joker="BJ"))
        return deck

    def _deal_100_plus_8(self) -> None:
        deck = self._build_deck()
        random.shuffle(deck)
        all_dealt = deck[:100]
        self.bottom = deck[100:108]
        for i in range(4):
            self.hands[i] = sorted(all_dealt[i * 25 : (i + 1) * 25], key=lambda c: self.card_sort_key(c))

    # ---------- 亮主 / 反主 ----------
    def _count_hearts5(self, hand: Sequence[Card]) -> int:
        return sum(1 for c in hand if (not c.joker and c.suit == "♥" and c.rank == "5"))

    def _ai_bid(self, pid: int, current: Optional[BidAction]) -> Optional[BidAction]:
        hand = self.hands[pid]
        hearts5 = self._count_hearts5(hand)
        suit_strength = {s: 0 for s in SUITS}
        for c in hand:
            if c.joker:
                continue
            if c.rank in {"A", "K", "Q", "J", "10"}:
                suit_strength[c.suit] += 2
            elif c.rank in {"2", "3"}:
                suit_strength[c.suit] += 3
            else:
                suit_strength[c.suit] += 1

        best_suit = max(suit_strength, key=suit_strength.get)
        best_score = suit_strength[best_suit]

        # 红桃5反主（最高）
        if hearts5 >= 2 and (current is None or current.level < 3):
            return BidAction(pid, "♥", 3)
        # 其他反主
        if current and current.level < 2 and best_score >= 15:
            return BidAction(pid, best_suit, 2)
        # 普通亮主
        if current is None and best_score >= 17:
            return BidAction(pid, best_suit, 1)
        return None

    def _parse_human_bid(self, txt: str, current: Optional[BidAction]) -> Optional[BidAction]:
        txt = txt.strip().upper()
        if txt == "PASS":
            return None
        # 格式：SHOW H / REBID S / H5
        if txt == "H5":
            if self._count_hearts5(self.hands[0]) >= 2 and (current is None or current.level < 3):
                return BidAction(0, "♥", 3)
            print("你没有两张红桃5，或当前已是同级/更高级亮主。")
            return None

        parts = txt.split()
        if len(parts) != 2 or parts[0] not in {"SHOW", "REBID"}:
            print("输入示例：SHOW H / REBID S / H5 / PASS")
            return None

        suit_map = {"S": "♠", "H": "♥", "C": "♣", "D": "♦"}
        if parts[1] not in suit_map:
            print("花色用 S/H/C/D")
            return None

        suit = suit_map[parts[1]]
        if parts[0] == "SHOW":
            if current is not None:
                print("已有亮主，只能反主（REBID）或H5。")
                return None
            return BidAction(0, suit, 1)

        # REBID
        if current is None:
            print("当前无人亮主，不能直接反主。")
            return None
        if current.level >= 2:
            print("当前已是反主级别，需H5才能再压。")
            return None
        return BidAction(0, suit, 2)

    def _bidding_phase(self) -> None:
        print("\n=== 亮主阶段 ===")
        print("你是P0。输入 PASS / SHOW H / REBID S / H5")
        current: Optional[BidAction] = None

        # 简化：按座次循环两轮，谁最终最高谁庄
        for _round in range(2):
            for pid in range(4):
                if pid == 0:
                    txt = input("亮主> ")
                    act = self._parse_human_bid(txt, current)
                else:
                    act = self._ai_bid(pid, current)

                if act is None:
                    if pid != 0:
                        print(f"P{pid} PASS")
                    continue

                if current is None or act.level > current.level:
                    current = act
                    print(f"P{pid} 亮主成功：{act.trump_suit}（级别{act.level}）")
                elif act.level == current.level and act.bidder != current.bidder:
                    # 同级后出覆盖前出（简化处理）
                    current = act
                    print(f"P{pid} 同级覆盖亮主：{act.trump_suit}")

                if current.level == 3:
                    break
            if current and current.level == 3:
                break

        if current is None:
            # 无人亮主：默认P0红桃主
            current = BidAction(0, "♥", 1)
            print("无人亮主，默认P0坐庄，主花色♥")

        self.dealer = current.bidder
        self.trump_suit = current.trump_suit
        self.dealer_team = TEAM_OF[self.dealer]
        self.attacker_team = 1 - self.dealer_team
        print(f"最终庄家: P{self.dealer}，主花色: {self.trump_suit}")

    # ---------- 排序/比较 ----------
    def card_sort_key(self, c: Card) -> Tuple[int, int]:
        return (0 if self.is_trump(c) else 1, -self.rank_value(c))

    def same_color_suit(self, suit: str) -> str:
        if suit == "♥":
            return "♦"
        if suit == "♦":
            return "♥"
        if suit == "♠":
            return "♣"
        return "♠"

    def is_trump(self, c: Card) -> bool:
        if c.joker:
            return True
        if c.suit == "♥" and c.rank == "5":
            return True
        if c.rank in {"2", "3"}:
            return True
        return c.suit == self.trump_suit

    def rank_value(self, c: Card) -> int:
        # 全局主牌序
        if not c.joker and c.suit == "♥" and c.rank == "5":
            return 1000
        if c.joker == "BJ":
            return 990
        if c.joker == "SJ":
            return 980

        # 3 的正副参谋
        if not c.joker and c.rank == "3":
            if c.suit == self.trump_suit:
                return 970
            if c.suit == self.same_color_suit(self.trump_suit):
                return 960
            return 950

        if not c.joker and c.rank == "2":
            if c.suit == self.trump_suit:
                return 940
            return 930

        # 主花色A..4
        face = {"A": 14, "K": 13, "Q": 12, "J": 11, "10": 10, "9": 9, "8": 8, "7": 7, "6": 6, "5": 5, "4": 4}
        if not c.joker and c.suit == self.trump_suit:
            return 800 + face[c.rank]

        # 副牌（无2/3）A..4
        if not c.joker:
            return face.get(c.rank, 0)

        return 0

    # ---------- 牌型 ----------
    def _pair_ranks_for_suit(self, cards: Sequence[Card], suit_like: str) -> List[int]:
        # suit_like: "TRUMP" or specific suit
        if suit_like == "TRUMP":
            pool = [c for c in cards if self.is_trump(c)]
        else:
            pool = [c for c in cards if (not self.is_trump(c) and c.suit == suit_like)]
        cnt = Counter((str(self.rank_value(c))) for c in pool)
        vals = [int(v) for v, n in cnt.items() if n >= 2]
        return sorted(vals)

    def detect_pattern(self, cards: List[Card]) -> Optional[Pattern]:
        n = len(cards)
        if n == 0:
            return None
        if n == 1:
            return Pattern("single", cards)

        vals = [self.rank_value(c) for c in cards]
        cnt = Counter(vals)

        if n == 2 and len(cnt) == 1:
            return Pattern("pair", cards)
        if n == 3 and len(cnt) == 1:
            return Pattern("triple", cards)
        if n == 4 and len(cnt) == 1:
            return Pattern("quad", cards)

        # 拖拉机：偶数张，>=4，且是连续对子，同一花色域（同主/同副）
        if n >= 4 and n % 2 == 0:
            # 全主 or 全同副花色
            all_trump = all(self.is_trump(c) for c in cards)
            all_same_sub = (not any(self.is_trump(c) for c in cards)) and len({c.suit for c in cards}) == 1
            if all_trump or all_same_sub:
                if all(v == 2 for v in cnt.values()):
                    seq = sorted(cnt.keys())
                    if all(seq[i + 1] - seq[i] == 1 for i in range(len(seq) - 1)):
                        return Pattern("tractor", cards)
        return None

    # ---------- 跟牌约束 ----------
    def lead_domain(self, lead_cards: List[Card]) -> str:
        if all(self.is_trump(c) for c in lead_cards):
            return "TRUMP"
        return lead_cards[0].suit

    def can_follow_pattern(self, hand: List[Card], lead: Pattern, domain: str, played: List[Card]) -> bool:
        # 必须同域同型尽力而为
        if domain == "TRUMP":
            domain_cards = [c for c in hand if self.is_trump(c)]
        else:
            domain_cards = [c for c in hand if (not self.is_trump(c) and c.suit == domain)]

        if not domain_cards:
            return True

        if lead.kind == "single":
            return len(played) == 1 and (self.is_trump(played[0]) if domain == "TRUMP" else (not self.is_trump(played[0]) and played[0].suit == domain))

        # pair/triple/quad 必须有对应张数同值，否则尽量出域内
        vals = Counter(self.rank_value(c) for c in domain_cards)
        need = {"pair": 2, "triple": 3, "quad": 4}.get(lead.kind, 0)
        if need:
            have = any(v >= need for v in vals.values())
            if have:
                if len(played) != need:
                    return False
                p = Counter(self.rank_value(c) for c in played)
                in_domain = all(self.is_trump(c) for c in played) if domain == "TRUMP" else all((not self.is_trump(c) and c.suit == domain) for c in played)
                return in_domain and len(p) == 1
            # 无法同型：必须尽可能出域内牌（简化为出任意同域数量）
            return all(self.is_trump(c) for c in played) if domain == "TRUMP" else all((not self.is_trump(c) and c.suit == domain) for c in played)

        if lead.kind == "tractor":
            pair_vals = self._pair_ranks_for_suit(domain_cards, domain)
            need_pairs = len(lead.cards) // 2
            can_make = False
            for i in range(len(pair_vals) - need_pairs + 1):
                if all(pair_vals[i + j + 1] - pair_vals[i + j] == 1 for j in range(need_pairs - 1)):
                    can_make = True
                    break

            pat = self.detect_pattern(played)
            if can_make:
                if not pat or pat.kind != "tractor" or len(played) != len(lead.cards):
                    return False
                # 同域
                if domain == "TRUMP":
                    return all(self.is_trump(c) for c in played)
                return all((not self.is_trump(c) and c.suit == domain) for c in played)
            # 做不出拖拉机，仍需尽量同域
            return all(self.is_trump(c) for c in played) if domain == "TRUMP" else all((not self.is_trump(c) and c.suit == domain) for c in played)

        return True

    def compare_play(self, lead: Pattern, a: List[Card], b: List[Card], domain: str) -> int:
        """比较a与b，返回1表示a大，-1表示b大，0平（不应出现）。"""
        # 同轮默认同型同长度（至少领出者一致）
        pa, pb = self.detect_pattern(a), self.detect_pattern(b)
        if not pa or not pb:
            return 0

        def is_domain(cards: List[Card], dom: str) -> bool:
            return all(self.is_trump(c) for c in cards) if dom == "TRUMP" else all((not self.is_trump(c) and c.suit == dom) for c in cards)

        a_dom = is_domain(a, domain)
        b_dom = is_domain(b, domain)

        # 同域优先
        if a_dom and not b_dom:
            return 1
        if b_dom and not a_dom:
            return -1

        # 均不同域：主牌毙副牌
        a_trump = all(self.is_trump(c) for c in a)
        b_trump = all(self.is_trump(c) for c in b)
        if a_trump and not b_trump:
            return 1
        if b_trump and not a_trump:
            return -1

        # 同类型比较主值
        va = max(self.rank_value(c) for c in a)
        vb = max(self.rank_value(c) for c in b)
        return 1 if va > vb else (-1 if vb > va else 0)

    # ---------- 计分 ----------
    def card_points(self, c: Card) -> int:
        if c.joker:
            return 0
        if c.rank == "5":
            return 5
        if c.rank in {"10", "K"}:
            return 10
        return 0

    def trick_points(self, cards: Iterable[Card]) -> int:
        return sum(self.card_points(c) for c in cards)

    # ---------- 输入与AI ----------
    def show_hand(self, pid: int) -> str:
        return "  ".join(f"[{i+1}:{c}]" for i, c in enumerate(self.hands[pid]))

    def choose_cards_by_index(self, pid: int, txt: str) -> Optional[List[Card]]:
        txt = txt.strip().upper()
        if txt == "PASS":
            return []
        try:
            idx = [int(x) - 1 for x in txt.split()]
        except ValueError:
            return None
        if not idx:
            return None
        hand = self.hands[pid]
        if any(i < 0 or i >= len(hand) for i in idx):
            return None
        out = [hand[i] for i in idx]
        return out

    def remove_cards(self, pid: int, cards: List[Card]) -> None:
        for c in cards:
            self.hands[pid].remove(c)

    def simple_ai_play(self, pid: int, lead: Optional[Pattern], domain: Optional[str]) -> List[Card]:
        hand = self.hands[pid]
        # 生成候选：单/对/三/四/拖拉机（最小可行）
        candidates: List[List[Card]] = []
        candidates.extend([[c] for c in hand])

        val_groups: Dict[int, List[Card]] = {}
        for c in hand:
            val_groups.setdefault(self.rank_value(c), []).append(c)
        for g in val_groups.values():
            if len(g) >= 2:
                candidates.append(g[:2])
            if len(g) >= 3:
                candidates.append(g[:3])
            if len(g) >= 4:
                candidates.append(g[:4])

        # very simple tractor
        for dom in ["TRUMP", "♠", "♥", "♣", "♦"]:
            pairs = self._pair_ranks_for_suit(hand, dom)
            if len(pairs) >= 2:
                seq = []
                for i in range(len(pairs) - 1):
                    if pairs[i + 1] - pairs[i] == 1:
                        seq = [pairs[i], pairs[i + 1]]
                        break
                if seq:
                    take: List[Card] = []
                    for v in seq:
                        got = [c for c in hand if self.rank_value(c) == v and ((dom == "TRUMP" and self.is_trump(c)) or (dom != "TRUMP" and (not self.is_trump(c) and c.suit == dom)))][:2]
                        take.extend(got)
                    if len(take) == 4:
                        candidates.append(take)

        # 过滤合法牌型
        legal = [c for c in candidates if self.detect_pattern(c)]
        legal.sort(key=lambda x: (len(x), max(self.rank_value(c) for c in x)))

        if lead is None:
            return legal[0]

        # 跟牌：先找同型同长度可跟且最小压制
        better = []
        for c in legal:
            p = self.detect_pattern(c)
            if not p:
                continue
            if p.kind != lead.kind or len(c) != len(lead.cards):
                continue
            if domain and not self.can_follow_pattern(hand, lead, domain, c):
                continue
            if self.compare_play(lead, c, lead.cards, domain or "TRUMP") == 1:
                better.append(c)

        if better:
            better.sort(key=lambda x: max(self.rank_value(i) for i in x))
            return better[0]

        # 压不住则尽量合规垫最小
        for c in legal:
            if domain and self.can_follow_pattern(hand, lead, domain, c):
                return c
        return [hand[0]]

    # ---------- 底牌 ----------
    def _dealer_take_and_discard_bottom(self) -> None:
        print(f"\n庄家P{self.dealer} 获得底牌8张: {' '.join(str(c) for c in self.bottom)}")
        self.hands[self.dealer].extend(self.bottom)
        self.hands[self.dealer].sort(key=lambda c: self.card_sort_key(c))

        if self.dealer == 0:
            while True:
                print("你的手牌(33张):")
                print(self.show_hand(0))
                raw = input("请选择8张扣底（输入8个序号）> ").strip()
                picked = self.choose_cards_by_index(0, raw)
                if not picked or len(picked) != 8:
                    print("需要准确输入8张序号。")
                    continue
                for c in picked:
                    self.hands[0].remove(c)
                self.bottom = picked
                break
        else:
            # AI 庄家：扣最小8张
            self.hands[self.dealer].sort(key=lambda c: self.rank_value(c))
            self.bottom = self.hands[self.dealer][:8]
            self.hands[self.dealer] = self.hands[self.dealer][8:]

        print(f"扣底完成。当前底牌: {' '.join(str(c) for c in self.bottom)}")

    # ---------- 主流程 ----------
    def play(self) -> None:
        trick_no = 1
        while any(len(self.hands[p]) > 0 for p in range(4)):
            print(f"\n====== 第{trick_no}墩，领出者 P{self.current_leader} ======")
            table: Dict[int, List[Card]] = {}
            order = [(self.current_leader + i) % 4 for i in range(4)]
            lead_pattern: Optional[Pattern] = None
            domain: Optional[str] = None

            for turn_idx, pid in enumerate(order):
                hand = self.hands[pid]
                if pid == 0:
                    print(f"\n你的手牌({len(hand)}):")
                    print(self.show_hand(0))
                    if lead_pattern:
                        print(f"领出牌型: {lead_pattern.kind}, 域: {domain}")
                    raw = input("出牌（序号，空格分隔）> ")
                    chosen = self.choose_cards_by_index(pid, raw)
                    if chosen is None or chosen == []:
                        print("本局不允许PASS，必须出牌。")
                        return
                else:
                    chosen = self.simple_ai_play(pid, lead_pattern, domain)

                pat = self.detect_pattern(chosen)
                if not pat:
                    if pid == 0:
                        print("非法牌型（本版本未开放甩牌）。")
                    else:
                        print(f"P{pid} AI出牌异常，自动改出最小单张")
                        chosen = [self.hands[pid][0]]
                        pat = Pattern("single", chosen)

                if turn_idx == 0:
                    lead_pattern = pat
                    domain = self.lead_domain(chosen)
                else:
                    assert lead_pattern and domain
                    if not self.can_follow_pattern(hand, lead_pattern, domain, chosen):
                        if pid == 0:
                            print("未按规则跟牌（需同域同型尽力）。")
                            return
                        # AI兜底：出最小同域单张
                        domain_cards = [c for c in hand if (self.is_trump(c) if domain == "TRUMP" else (not self.is_trump(c) and c.suit == domain))]
                        chosen = [domain_cards[0]] if domain_cards else [hand[0]]
                        pat = Pattern("single", chosen)

                self.remove_cards(pid, chosen)
                table[pid] = chosen
                who = "你" if pid == 0 else f"P{pid}"
                print(f"{who} 出: {' '.join(str(c) for c in chosen)}")

            # 比大小
            winner = order[0]
            for pid in order[1:]:
                cmp = self.compare_play(lead_pattern, table[pid], table[winner], domain or "TRUMP")
                if cmp == 1:
                    winner = pid

            # 本墩得分
            trick_cards = [c for pid in order for c in table[pid]]
            points = self.trick_points(trick_cards)
            self.scores[TEAM_OF[winner]] += points
            print(f"本墩赢家: P{winner}，得分 {points}，当前队伍分 Team0={self.scores[0]} Team1={self.scores[1]}")

            self.last_trick_winner = winner
            self.last_trick_pattern = lead_pattern.kind if lead_pattern else "single"
            self.current_leader = winner
            trick_no += 1

            # 结束条件：有人出完即最后一墩完成后结束
            if any(len(self.hands[p]) == 0 for p in range(4)):
                if all(len(self.hands[p]) == 0 for p in range(4)):
                    break
                if sum(len(self.hands[p]) for p in range(4)) == 0:
                    break

            if sum(len(self.hands[p]) for p in range(4)) == 0:
                break

        self.apply_koudi()
        self.show_result()

    def apply_koudi(self) -> None:
        bottom_pts = self.trick_points(self.bottom)
        mul = 2
        if self.last_trick_pattern == "pair":
            mul = 4
        elif self.last_trick_pattern == "tractor":
            mul = 8
        gain = bottom_pts * mul
        self.scores[TEAM_OF[self.last_trick_winner]] += gain
        print(f"\n抠底：底牌分 {bottom_pts} x{mul} = {gain}，计入 Team{TEAM_OF[self.last_trick_winner]}")

    def show_result(self) -> None:
        idle_team = self.attacker_team
        print("\n===== 对局结束 =====")
        print(f"庄家: P{self.dealer} (Team{self.dealer_team})")
        print(f"队伍总分：Team0={self.scores[0]} Team1={self.scores[1]}")
        if self.scores[idle_team] >= 80:
            print(f"闲家队 Team{idle_team} 胜（>=80）")
        else:
            print(f"庄家队 Team{self.dealer_team} 胜（闲家<{80}）")


def main() -> None:
    print("六安红桃5（对门一班）启动。玩家编号固定：你=P0，AI为P1/P2/P3")
    g = LuAnHongTao5()
    g.play()


if __name__ == "__main__":
    main()
