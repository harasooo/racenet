import dataclasses
import random
from collections import Counter
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.stats import norm


class PlayerRelType(BaseModel):
    type_id: int
    effective: Optional[int]


class PlayerAbsType(BaseModel):
    type_num: int
    type_id: int
    t: float
    tmp_v: float

    def update(self, n_race: int):
        if n_race < (self.type_id * self.t) / self.type_num:
            self.tmp_v = n_race * self.type_num / (self.type_id * self.t)
        else:
            tmp_a = 1 / ((self.type_id / (self.type_num * self.t)) - self.t)
            tmp_b = -tmp_a * self.t
            self.tmp_v = n_race * tmp_a + tmp_b


class Player(BaseModel):
    p_id: int
    abs_ability: float
    p_rel_type: PlayerRelType
    p_abs_type: PlayerAbsType
    n_race: int = 0

    def chenge_rel_type(self, effective):
        self.p_rel_type.effective = effective

    def update_abs_type(self, effective):
        self.p_rel_type.effective = effective

    def __hash__(self):
        return hash((self.p_id))

    def __eq__(self, other):
        if type(other) == Player:
            return (self.p_id == other.p_id)
        else:
            return False


class RaceResultData(BaseModel):
    score: float
    rank: int


class RaceResult(BaseModel):
    t_id: int
    results: List[Dict[Player, RaceResultData]]


@dataclasses.dataclass
class Race:
    player_set: Set[Player]
    n_players: int
    stochastic: bool
    random_sampling: bool
    history: List[RaceResult] = dataclasses.field(default_factory=list)
    df_cols = ["race_id", "player_id", "abs",
               "p_rel_type", "effective", "score", "rank"]
    history_df: pd.DataFrame = pd.DataFrame(index=[], columns=df_cols)
    tmp_t: int = 0

    def _calc_type_coef_list(self, tmp_players: List[Player]):
        type_dic = Counter(tmp_players)
        type_coef_list = []
        for p in tmp_players:
            if p.p_rel_type.effective:
                n_eff = type_dic[type_dic[p.p_rel_type.effective]]
                type_coef_list.append(self._type_coef(n_eff))
            else:
                type_coef_list.append(0)
        return type_coef_list

    @staticmethod
    def _type_coef(n_eff: int):
        if n_eff == 0:
            return norm.ppf(0.5, 0, 2)
        elif n_eff == 1:
            return norm.ppf(0.6, 0, 2)
        elif n_eff == 2:
            return norm.ppf(0.7, 0, 2)
        elif n_eff == 3:
            return norm.ppf(0.8, 0, 2)
        elif n_eff == 4:
            return norm.ppf(0.9, 0, 2)
        else:
            raise ValueError("you have to set specify one of [0,1,2,3,4]")

    @staticmethod
    def _score_to_rank(score_list):
        rank_list = [1 for _ in range(len(score_list))]
        for i, j in enumerate(score_list):
            for k in score_list:
                if j < k:
                    rank_list[i] += 1
        return rank_list

    @staticmethod
    def _random_sampling_players(player_set, n_players):
        return random.sample(player_set, n_players)

    def _biased_sampling_plyaers(player_set, n_players):
        pass

    def save_df_history(self, race_id, players, score, rank):
        race_s = pd.Series([race_id for _ in range(self.n_players)])
        player_s = pd.Series([p.p_id for p in players])
        abs_s = pd.Series([p.abs_ability for p in players])
        type_s = pd.Series([p.p_rel_type.type_id for p in players])
        eff_s = pd.Series([p.p_rel_type.effective for p in players])
        score_s = pd.Series(score)
        rank_s = pd.Series(rank)
        temp_race_df = pd.concat(
            [race_s, player_s, abs_s, type_s, eff_s, score_s, rank_s], axis=1)
        temp_race_df.columns = self.df_cols
        self.history_df = pd.concat(
            [self.history_df, temp_race_df]).reset_index(drop=True)

    def do_race(self, n_times):
        for _ in range(n_times):
            # playerをサンプリング
            tmp_players = self._random_sampling_players(
                self.player_set, self.n_players)
            # 相性情報を取得
            type_coef_list = self._calc_type_coef_list(tmp_players)
            score_list = []
            # それぞれのプレイヤーのスコアを算出
            for p, type_coef in zip(tmp_players, type_coef_list):
                tmp_abs_ability = np.random.normal(p.abs_ability, 1)
                if self.stochastic:
                    tmp_score = np.random.normal(
                        tmp_abs_ability + type_coef, 1)
                else:
                    tmp_score = tmp_abs_ability + type_coef
                    p.n_race += 1
                score_list.append(tmp_score)
            # 各プレイヤーのスコアからランクを算出
            rank_list = self._score_to_rank(score_list)
            # 結果を記録
            self.save_df_history(self.tmp_t, tmp_players,
                                 score_list, rank_list)
            results = [{p: RaceResultData(score=s, rank=r)} for p, s, r in zip(
                tmp_players, score_list, rank_list)]
            self.history.append(RaceResult(t_id=self.tmp_t, results=results))
            self.tmp_t += 1


def make_player_set():
    player_list = []
    for i in range(20):
        abs_ability = np.random.normal(0, 1)
        type_id = np.random.choice([1, 2, 3], 1)
        eff_type_id = (type_id + 1) if type_id < 3 else 1
        player_list.append(Player(p_id=i, abs_ability=abs_ability, p_rel_type=PlayerRelType(
            type_id=type_id, effective=eff_type_id)))
    player_set = set(player_list)
    return player_set
