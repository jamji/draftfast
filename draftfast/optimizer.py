from collections import defaultdict
from typing import List
from ortools.linear_solver import pywraplp
from draftfast.settings import OptimizerSettings
from draftfast.dke_exceptions import (InvalidBoundsException,
                                      PlayerBanAndLockException)
from draftfast.orm import Player
from draftfast.rules import RuleSet, DRAFT_KINGS
from draftfast.lineup_constraints import LineupConstraints


class Optimizer(object):
    def __init__(
        self,
        players: List[Player],
        rule_set: RuleSet,
        settings: OptimizerSettings,
        lineup_constraints: LineupConstraints,
        exposure_dict: dict
    ):
        self.solver = pywraplp.Solver(
            'FD',
            pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
        )
        self.players = players
        self.enumerated_players = list(enumerate(players))
        self.existing_rosters = settings.existing_rosters or []
        self.salary_min = rule_set.salary_min
        self.salary_max = rule_set.salary_max
        self.roster_size = rule_set.roster_size
        self.position_limits = rule_set.position_limits
        self.offensive_positions = rule_set.offensive_positions
        self.defensive_positions = rule_set.defensive_positions
        self.general_position_limits = rule_set.general_position_limits
        self.showdown = rule_set.game_type == 'showdown'
        self.single = rule_set.game_type == 'single'
        self.flex3 = rule_set.game_type == 'flex3'
        self.is_draftkings = rule_set.site == DRAFT_KINGS
        self.flexy_five = rule_set.game_type == 'flexy_five'
        self.settings = settings
        self.lineup_constraints = lineup_constraints
        self.banned_for_exposure = exposure_dict.get('banned', [])
        self.locked_for_exposure = exposure_dict.get('locked', [])

        self.name_to_idx_map = {}
        self.variables = []
        self.name_to_idx_map = dict()
        if self.single or self.flex3:
            self.player_to_idx_map = dict()
        else:
            self.player_to_idx_map = defaultdict(list)

        for idx, player in self.enumerated_players:
            self.variables.append(
                self.solver.IntVar(0, 1, player.solver_id)
            )

            self._add_player_to_idx_maps(player, idx)

            if self._is_locked(player):
                player.lock = True
            if self._is_banned(player):
                player.ban = True
            if self._is_position_locked(player):
                player.position_lock = True
            if self._is_position_banned(player):
                player.position_ban = True

            # TODO: this can only happen because of exposure, but it could be
            # handled better
            if player.lock and player.ban:
                raise PlayerBanAndLockException(player.name)

        self.teams = set([p.team for p in self.players])
        self.names = set([p.name for p in self.players])
        self.objective = self.solver.Objective()
        self.objective.SetMaximization()

    def _add_player_to_idx_maps(self, p: Player, idx: int):
        if self.single or self.flex3:
            self.player_to_idx_map[p.solver_id] = idx
        else:
            self.player_to_idx_map[p.solver_id.split('-')[0]].append(idx)

        if p.name not in self.name_to_idx_map.keys():
            self.name_to_idx_map[p.name] = set()
        self.name_to_idx_map[p.name].update([idx])

    def _is_locked(self, p: Player) -> bool:
        return self.lineup_constraints.is_locked(p.name) or \
               p.name in self.locked_for_exposure or \
               p.lock

    def _is_banned(self, p: Player) -> bool:
        return self.lineup_constraints.is_banned(p.name) or \
               p.name in self.banned_for_exposure or \
               p.ban

    def _is_position_locked(self, p: Player) -> bool:
        return self.lineup_constraints.is_position_locked(p.solver_id)

    def _is_position_banned(self, p: Player) -> bool:
        return self.lineup_constraints.is_position_banned(p.solver_id)

    def solve(self) -> bool:
        self._set_player_constraints()
        self._set_player_group_constraints()
        self._optimize_on_projected_points()
        self._set_salary_range()
        self._set_roster_size()
        self._set_positions()
        self._set_general_positions()
        self._set_stack()
        self._set_combo()
        self._set_no_duplicate_lineups()
        if not self.flexy_five:
            self._set_min_teams()
            self._set_max_players_per_team()
        self._set_po_settings()

        if self.single or self.flex3:
            self._set_no_duplicate_players()

        if self.offensive_positions and self.defensive_positions \
                and self.settings.no_offense_against_defense or \
                self.showdown and self.settings.no_defense_against_captain:
            self._set_no_opp_defense()

        solution = self.solver.Solve()

        return solution == self.solver.OPTIMAL

    def _set_player_constraints(self):
        multi_constraints = dict()

        for i, p in self.enumerated_players:
            lb = 1 if (p.lock or p.position_lock) else 0
            ub = 0 if (p.ban or p.position_ban) else 1

            if lb > ub:
                raise InvalidBoundsException

            if (p.multi_position or self.showdown) and not (
                    p.position_lock or p.position_ban):
                if p.name not in multi_constraints.keys():
                    multi_constraints[p.name] = self.solver.Constraint(lb, ub)
                constraint = multi_constraints[p.name]
            elif (p.multi_position or self.showdown) and p.position_lock:
                if p.name not in multi_constraints.keys():
                    multi_constraints[p.name] = self.solver.Constraint(0, ub)
                multi_constraints[p.name].SetCoefficient(self.variables[i], 1)

                constraint = self.solver.Constraint(lb, ub)
            else:
                constraint = self.solver.Constraint(lb, ub)

            constraint.SetCoefficient(self.variables[i], 1)

    def _set_player_group_constraints(self):
        for group_constraint in self.lineup_constraints:
            if group_constraint.exact:
                lb = ub = group_constraint.exact
            else:
                lb = group_constraint.lb
                ub = group_constraint.ub

            constraint = self.solver.Constraint(lb, ub)
            for name in group_constraint.players:
                for idx in self.name_to_idx_map[name]:
                    constraint.SetCoefficient(self.variables[idx], 1)

    def _optimize_on_projected_points(self):
        for i, player in self.enumerated_players:
            self.objective.SetCoefficient(
                self.variables[i],
                player.proj,
            )

    def _set_salary_range(self):
        salary_cap = self.solver.Constraint(
            self.salary_min,
            self.salary_max,
        )
        for i, player in self.enumerated_players:
            salary_cap.SetCoefficient(
                self.variables[i],
                player.cost
            )

    def _set_roster_size(self):
        size_cap = self.solver.Constraint(
            self.roster_size,
            self.roster_size,
        )

        for variable in self.variables:
            size_cap.SetCoefficient(variable, 1)

    def _set_stack(self):
        if self.settings:
            stacks = self.settings.stacks

            if stacks:
                for stack in stacks:
                    stack_team = stack.team
                    stack_count = stack.count
                    stack_cap = self.solver.Constraint(
                        stack_count,
                        stack_count,
                    )

                    for i, player in self.enumerated_players:
                        if stack_team == player.team:
                            stack_cap.SetCoefficient(
                                self.variables[i],
                                1
                            )

    def _set_combo(self):
        if self.settings:
            combo = self.settings.force_combo
            combo_allow_te = self.settings.combo_allow_te

            combo_skill_type = ['WR']
            if combo_allow_te:
                combo_skill_type.append('TE')

            if combo:
                teams = set([p.team for p in self.players])
                enumerated_players = self.enumerated_players

                for team in teams:
                    skillplayers_on_team = [
                        self.variables[i] for i, p in enumerated_players
                        if p.team == team and p.pos in combo_skill_type
                    ]
                    qbs_on_team = [
                        self.variables[i] for i, p in enumerated_players
                        if p.team == team and p.pos == 'QB'
                    ]
                    self.solver.Add(
                        self.solver.Sum(skillplayers_on_team) >=
                        self.solver.Sum(qbs_on_team)
                    )

    def _set_no_opp_defense(self):
        offensive_pos = self.offensive_positions
        defensive_pos = self.defensive_positions

        enumerated_players = self.enumerated_players

        for team in self.teams:
            offensive_against = [
                self.variables[i] for i, p in enumerated_players
                if p.pos in offensive_pos and
                p.is_opposing_team_in_match_up(team)
            ]

            # TODO this is gross for showdown
            defensive = [
                self.variables[i] for i, p in enumerated_players
                if p.team == team and p.pos in defensive_pos or
                self.showdown and p.real_pos in defensive_pos
            ]

            for p in offensive_against:
                for d in defensive:
                    self.solver.Add(p <= 1 - d)

    def _set_positions(self):
        for position, min_limit, max_limit in self.position_limits:
            position_cap = self.solver.Constraint(
                min_limit,
                max_limit
            )

            for i, player in self.enumerated_players:
                if position == player.pos:
                    position_cap.SetCoefficient(self.variables[i], 1)

    def _set_general_positions(self):
        for general_position, min_limit, max_limit in \
                self.general_position_limits:
            position_cap = self.solver.Constraint(min_limit, max_limit)

            for i, player in self.enumerated_players:
                if general_position == player.nba_general_position:
                    position_cap.SetCoefficient(
                        self.variables[i],
                        1
                    )

    def _set_no_duplicate_lineups(self):
        for roster in self.existing_rosters:
            max_repeats = self.roster_size
            if self.settings.uniques:
                max_repeats = max(
                    self.roster_size - self.settings.uniques,
                    0
                )
            repeated_players = self.solver.Constraint(
                0,
                max_repeats
            )
            for player in roster.sorted_players():
                if self.single or self.flex3:
                    i = self.player_to_idx_map.get(player.solver_id)
                    if i is not None:
                        repeated_players.SetCoefficient(self.variables[i], 1)
                else:
                    indexes = self.player_to_idx_map.get(player.solver_id.split('-')[0])
                    if indexes is not None:
                        for i in indexes:
                            repeated_players.SetCoefficient(self.variables[i], 1)

    def _set_min_teams(self):
        teams = []

        for team in self.teams:
            if team:
                team_var = self.solver.IntVar(0, 1, team)
                teams.append(team_var)
                players_on_team = [
                    self.variables[i] for i, p
                    in self.enumerated_players if p.team == team
                ]
                self.solver.Add(team_var <= self.solver.Sum(players_on_team))

        # TODO - add constraint of max players per team per sport
        if len(teams) > 0:
            self.solver.Add(
                self.solver.Sum(teams) >= self.settings.min_teams
            )

    def _set_no_duplicate_players(self):
        """Single game and Flex3, a player will be 4 (3) variable"""
        for name in self.names:
            name_var = self.solver.IntVar(0, 1, name)
            players_on_name = [self.variables[i] for i, p in self.enumerated_players if p.name == name]
            self.solver.Add(self.solver.Sum(players_on_name) <= name_var)

    def _set_max_players_per_team(self):
        max_players_per_team = 4
        if self.is_draftkings:
            max_players_per_team = 7
        for team in self.teams:
            if team:
                if self.single:
                    team_cap = self.solver.Constraint(1, 4)
                elif self.flex3:
                    team_cap = self.solver.Constraint(0, 2)
                else:
                    team_cap = self.solver.Constraint(0, max_players_per_team)
                for i, player in self.enumerated_players:
                    if team == player.team:
                        team_cap.SetCoefficient(self.variables[i], 1)

    def _set_po_settings(self):
        for po_setting in self.settings.lineup_settings:
            player_lower_bound = po_setting['playerCount']
            po_upper_bound = po_setting['poUpperBound']
            po_cap = self.solver.Constraint(player_lower_bound, self.roster_size)
            for i, player in self.enumerated_players:
                if player.po < po_upper_bound:
                    po_cap.SetCoefficient(
                        self.variables[i],
                        1
                    )
