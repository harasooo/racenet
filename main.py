import dataclasses
import random
from typing import Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.stats import norm


class PlayerRelType(BaseModel):
    type_id: int


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
    is_relative: bool
    eff_dic: Dict[int, int]
    change_player_abs_type: bool
    change_player_rel_type: bool
    change_player_rel_type_freq: int
    change_player_rel_type_ratio: float
    history: List[RaceResult] = dataclasses.field(default_factory=list)
    df_cols = ["race_id", "player_id", "abs",
               "p_rel_type", "effective", "score", "rank"]
    history_df: pd.DataFrame = pd.DataFrame(index=[], columns=df_cols)
    tmp_t: int = 0

    def _calc_type_coef_list(self, tmp_players: List[Player]):

        type_coef_list = []
        for p in tmp_players:
            p_eff = self.eff_dic[p.p_rel_type.type_id]
            n_eff = sum([1 if op.p_rel_type.type_id ==
                        p_eff else 0 for op in tmp_players])
            type_coef_list.append(self._type_coef(n_eff))
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
        elif n_eff == 5:
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

    def _biased_sampling_plyaers(self, player_set: Set[Player], n_players: int, biased_ratio: float, biased_type: int, uneff_n: int):
        if random.random() < biased_ratio:
            return random.sample(player_set, n_players)
        else:
            # まず基準となるplayerを選択
            flg_1 = True
            while flg_1:
                p_list: List[Player] = random.sample(player_set, 1)
                if p_list[0].p_rel_type.type_id == biased_type:
                    if p_list[0].abs_ability > norm.ppf(0.75, 0, 1):
                        flg_1 = False
            # uneff_nの数だけドロー
            flg_2 = 0
            while flg_2 < uneff_n:
                o_p: Player = random.sample(player_set, 1)[0]
                if o_p.p_id not in p_list:
                    if self.eff_dic[o_p.p_rel_type.type_id] == biased_type:
                        p_list.append(o_p)
                        flg_2 += 1
            remain_n = n_players - (uneff_n + 1)
            remain_player_set = player_set - set(p_list)
            return p_list + random.sample(remain_player_set, remain_n)

    def _change_eff_dic(self):
        self.eff_dic = {k: v for k, v in zip(self.eff_dic.keys(),
                                             random.sample(list(self.eff_dic.values()), len(list(self.eff_dic.values()))))}

    def save_df_history(self, race_id, players, score, rank):
        race_s = pd.Series([race_id for _ in range(self.n_players)])
        player_s = pd.Series([p.p_id for p in players])
        abs_s = pd.Series([p.abs_ability for p in players])
        type_s = pd.Series([p.p_rel_type.type_id for p in players])
        eff_s = pd.Series([self.eff_dic[p.p_rel_type.type_id]
                          for p in players])
        score_s = pd.Series(score)
        rank_s = pd.Series(rank)
        temp_race_df = pd.concat(
            [race_s, player_s, abs_s, type_s, eff_s, score_s, rank_s], axis=1)
        temp_race_df.columns = self.df_cols
        self.history_df = pd.concat(
            [self.history_df, temp_race_df]).reset_index(drop=True)

    def do_race(self, n_times, switch_biased_samplimg_ration: Union[bool, int] = False, biased_ratio: float = 0, biased_type: int = 0, neff_n: int = 0):
        for n in range(n_times):
            # 相性変化
            # 一定の周期で判定
            if self.change_player_rel_type:
                if n % self.change_player_rel_type_freq == 0:
                    if random.random() < self.change_player_rel_type_ratio:
                        self._change_eff_dic()
            # playerをサンプリング
            if switch_biased_samplimg_ration:
                if switch_biased_samplimg_ration < n:
                    tmp_players = self._biased_sampling_plyaers(
                        self.player_set, self.n_players, biased_ratio, biased_type, neff_n)
                else:
                    tmp_players = self._random_sampling_players(
                        self.player_set, self.n_players)
            else:
                tmp_players = self._random_sampling_players(
                    self.player_set, self.n_players)
            # 相性情報を取得
            if self.is_relative:
                type_coef_list = self._calc_type_coef_list(tmp_players)
            else:
                type_coef_list = [0 for i in range(len(tmp_players))]
            score_list = []
            # それぞれのプレイヤーのスコアを算出
            for p, type_coef in zip(tmp_players, type_coef_list):
                # 時間変化（絶対的な能力）
                if self.change_player_abs_type:
                    p.p_abs_type.update(p.n_race)
                    tmp_p_abs_v = p.p_abs_type.tmp_v
                else:
                    tmp_p_abs_v = 0
                # 確率的かどうか
                if self.stochastic:
                    tmp_abs_ability = np.random.normal(
                        p.abs_ability + tmp_p_abs_v, 1)
                    tmp_score = np.random.normal(
                        tmp_abs_ability + type_coef, 1)
                else:
                    tmp_abs_ability = p.abs_ability
                    tmp_score = tmp_abs_ability + type_coef + tmp_p_abs_v
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


def make_player_set(n_players: int, rel_n_type: int, abs_n_type: int):
    player_list = []
    for i in range(n_players):
        abs_ability = np.random.normal(0, 1)
        abs_type_id = np.random.choice([i+1 for i in range(abs_n_type)], 1)
        # rel type
        rel_type_id = np.random.choice([i+1 for i in range(rel_n_type)], 1)
        eff_type_id = (rel_type_id + 1) if rel_type_id < rel_n_type else 1
        player_list.append(Player(p_id=i, abs_ability=abs_ability,
                                  p_abs_type=PlayerAbsType(
                                      type_num=4, type_id=abs_type_id, t=200, tmp_v=0),
                                  p_rel_type=PlayerRelType(type_id=rel_type_id, effective=eff_type_id)))
    player_set = set(player_list)
    return player_set
