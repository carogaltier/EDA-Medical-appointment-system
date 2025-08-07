# =====================================================================================
# Synthetic Medical Appointment Scheduler
# =====================================================================================
# Description:
# This module defines the `AppointmentScheduler` class, which generates realistic,
# synthetic data for medical appointment systems. It simulates:
#   - Working slot schedules
#   - Patient demographics and turnover
#   - Appointment booking, cancellation, and rebooking behavior
#   - Arrival times, delays, durations, and attendance outcomes
#
# Intended for use in educational settings, data science portfolios, and prototyping.
#
# Dependencies: pandas, numpy, datetime, Faker
#
# -------------------------------------------------------------------------------------
# Author: Carolina Gonzalez Galtier
# Created: August 2025
# License: CC BY 4.0
# Repository: https://github.com/carogaltier/Synthetic-Medical-Appointment-Dataset
# =====================================================================================

# Libraries
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date, time
from typing import List, Tuple, Optional
from faker import Faker
#from .constants import (
#    DEFAULT_STATUS_RATES,
#    DEFAULT_FIRST_ATTENDANCE_RATIO,
#    DEFAULT_AGE_GENDER_PROBS
#)
from itertools import product



DEFAULT_AGE_GENDER_PROBS = [
    {"age_yrs": "0-4", "total_female": 0.01413, "total_male": 0.01794},
    {"age_yrs": "5-9", "total_female": 0.01140, "total_male": 0.01409},
    {"age_yrs": "10-14", "total_female": 0.01318, "total_male": 0.01459},
    {"age_yrs": "15-19", "total_female": 0.01738, "total_male": 0.01348},
    {"age_yrs": "20-24", "total_female": 0.02326, "total_male": 0.01010},
    {"age_yrs": "25-29", "total_female": 0.03988, "total_male": 0.01208},
    {"age_yrs": "30-34", "total_female": 0.05164, "total_male": 0.01449},
    {"age_yrs": "35-39", "total_female": 0.04369, "total_male": 0.01591},
    {"age_yrs": "40-44", "total_female": 0.03240, "total_male": 0.01754},
    {"age_yrs": "45-49", "total_female": 0.02902, "total_male": 0.01861},
    {"age_yrs": "50-54", "total_female": 0.03684, "total_male": 0.02513},
    {"age_yrs": "55-59", "total_female": 0.04172, "total_male": 0.03249},
    {"age_yrs": "60-64", "total_female": 0.04188, "total_male": 0.03723},
    {"age_yrs": "65-69", "total_female": 0.03939, "total_male": 0.03822},
    {"age_yrs": "70-74", "total_female": 0.04026, "total_male": 0.03995},
    {"age_yrs": "75-79", "total_female": 0.04395, "total_male": 0.04334},
    {"age_yrs": "80-84", "total_female": 0.03090, "total_male": 0.02876},
    {"age_yrs": "85-89", "total_female": 0.02015, "total_male": 0.01745},
    {"age_yrs": "90+", "total_female": 0.01040, "total_male": 0.00716},
]


DEFAULT_STATUS_RATES = {
    "attended": 0.773,
    "cancelled": 0.164,
    "did not attend": 0.059,
    "unknown": 0.004
}

DEFAULT_FIRST_ATTENDANCE_RATIO = 0.325

# -------------------------------------------------------------------------------------
# Default Parameter Estimates
# -------------------------------------------------------------------------------------
# The following default values are derived from NHS England Outpatient Statistics
# (2023–2024), available publicly at:
# https://www.england.nhs.uk/statistics/statistical-work-areas/outpatient-activity/

# These were manually extracted using the `parse_nhs_outpatient_data()` function
# from the optional module `parse_nhs_outpatient_data.py`, which processes the
# following summary sheets:
#   - Summary Report 1 → status distribution (DEFAULT_STATUS_RATES)
#   - Summary Report 2 → first attendance ratio (DEFAULT_FIRST_ATTENDANCE_RATIO)
#   - Summary Report 3 → age-gender distribution (DEFAULT_AGE_GENDER_PROBS)

# These values are included here as static constants for portability and to avoid
# runtime dependencies on external Excel files.
# -------------------------------------------------------------------------------------



class AppointmentScheduler:
    """
    Synthetic medical appointment scheduler and simulator.

    This class generates a realistic dataset of appointment slots, patients, and appointment records
    based on configurable parameters such as clinic hours, fill rates, patient turnover, and real-world
    attendance statistics. It is designed for educational, analytical, and prototyping use cases.

    Parameters
    ----------
    date_ranges : list of (datetime, datetime), optional
        List of date intervals during which appointments can be scheduled.
        Each tuple must be in the format (start, end) with start < end.
        Ranges must not overlap and must span at least one day.
        Default: [(2023-01-01, 2024-12-31)]

    ref_date : datetime, optional
        Reference date that splits past vs. future appointments.
        Must be a datetime object falling within the `date_ranges` period.
        If not provided, the last day of the latest date_range is used.
        If `ref_date + booking_horizon` exceeds the last range, it is extended.

    working_days : list of int, optional
        Weekdays when the clinic operates (0=Monday, ..., 6=Sunday).
        Must be a non-empty list of unique integers between 0 and 6.
        Default: [0, 1, 2, 3, 4]

    appointments_per_hour : int
        Number of bookable appointments per hour (determines slot duration).
        Must divide 60 evenly. Accepted values are: 1 (60 min), 2 (30 min), 3 (20 min),
        4 (15 min), 6 (10 min). Default: 4

    working_hours : list of (int, int), optional
        Time blocks of daily working hours (e.g., [(8, 12), (14, 18)]).
        Each segment must be between 0 and 24, with start < end.
        Segments must not overlap and must allow for at least one slot
        given the `appointments_per_hour`.
        Default: [(8, 18)]


    fill_rate : float
        Target proportion of available slots that are eventually booked and attended.
        Must be between 0.2 and 1.0. Values below 0.2 imply implausibly low utilization
        and are likely to yield unrealistic or unstable scheduling behavior.
        Default: 0.9

    booking_horizon : int
        Maximum number of days into the future that appointments can be scheduled.
        Must be an integer between 0 and 150 (approx. 5 months).
        If set to 0, only historical appointments will be simulated.
        Default: 30

    median_lead_time : int
        Median number of days between scheduling and appointment date.
        Must be at least 1 and cannot exceed the booking_horizon.
        Used to control the exponential distribution of lead times.
        Default: 10
        
    status_rates : dict, optional
        Dictionary specifying the probability distribution of appointment outcomes for past slots.
        Must contain exactly the following keys: {"attended", "cancelled", "did not attend", "unknown"}.
        Values must be non-negative and will be normalized to sum to 1.
        If not provided, defaults are derived from NHS outpatient statistics.
        Default: {"attended": 0.773, "cancelled": 0.164, "did not attend": 0.059, "unknown": 0.004}

    rebook_category : str
        Controls the intensity of rebooking behavior after cancellations.
        Must be one of: {"min", "med", "max"}, corresponding to 0%, 50%, or 100% of cancelled appointments being rescheduled.
        Default: "med"

    check_in_time_mean : float
        Average number of minutes before or after the scheduled time that patients arrive.
        Negative values indicate early arrivals (e.g., -10 means 10 minutes early).
        The variability is modeled using a fixed standard deviation of 9.8 minutes.
        Must be between -60 and +30. Default: -10

    visits_per_year : float
        Average number of outpatient visits per patient per year.
        Must be between 0.5 and 10. This value controls the volume of patients needed to fulfill all appointments.
        Default: 1.2

    first_attendance : float, optional
        Proportion of patients in each year who are first-time visitors (i.e., not previously seen).
        Must be a float between 0 and 1. If None, a default value derived from NHS statistics (0.325) is used.
        Default: None

    bin_size : int
        Width (in years) of age group intervals used for binning patients in age-based analyses.
        Must be an integer between 1 and 20. For example, bin_size=5 yields age groups like 15–19, 20–24, etc.
        Default: 5

    lower_cutoff : int
        Minimum patient age to include in the simulation.
        Patients younger than this age are excluded or ignored depending on the `truncated` flag.
        Must be less than `upper_cutoff`. Default: 15

    upper_cutoff : int
        Maximum patient age for inclusion in the cohort.
        Used when sampling ages from demographic distributions.
        Must be greater than `lower_cutoff`. Default: 90

    truncated : bool
        Whether to exclude all age bins below `lower_cutoff` from the age-gender distribution entirely.
        If True, patients under `lower_cutoff` are not generated at all.
        If False, the full distribution is used but ages are clipped at runtime.
        Default: True

    age_gender_probs : pandas.DataFrame, optional
        External age-sex probability table. If not provided, a default NHS-based distribution is used.
        The DataFrame must contain columns: "age_yrs", "total_male", "total_female".
        Default: None
    
    seed : int
        Random seed for reproducibility. Must be a non-negative integer.
        A fixed seed ensures the same synthetic dataset is generated every time.
        Default: 42

    noise : float
        Magnitude of random variation to introduce in patient generation,
        appointment timing, attendance, and rebooking behavior.
        Must be between 0 and 1. Default: 0.1

    """

    def __init__(self,
                 
        # ============================
        # 1. Scheduling configuration
        # ============================
        date_ranges: Optional[List[Tuple[datetime, datetime]]] = None,
        ref_date: Optional[datetime] = None,
        working_days: Optional[List[int]] = None,
        appointments_per_hour: int = 4,
        working_hours: Optional[List[Tuple[int, int]]] = None,
        fill_rate: float = 0.9,
        booking_horizon: int = 30,
        median_lead_time: int = 10,

        # ============================
        # 2. Attendance behavior and appointment outcomes
        # ============================
        status_rates: Optional[dict] = None,
        rebook_category: str = "med",
        check_in_time_mean: float = -10,

        # ============================
        # 3. Patient visit patterns and turnover
        # ============================
        visits_per_year: float = 1.2,
        first_attendance: Optional[float] = None,

        # ============================
        # 4. Demographic configuration
        # ============================
        bin_size: int = 5,
        lower_cutoff: int = 15,
        upper_cutoff: int = 90,
        truncated: bool = True,
        age_gender_probs: Optional[pd.DataFrame] = None,

        # ============================
        # 5. Reproducibility and stochasticity
        # ============================
        seed: int = 42,
        noise: float = 0.1,
    ):   
        

        # ============================
        # VALIDATION: date_ranges
        # ============================
        if date_ranges is None:
            self.date_ranges = [(datetime(2023, 1, 1), datetime(2024, 12, 31))]
        else:
            if not isinstance(date_ranges, list) or not all(isinstance(pair, tuple) and len(pair) == 2 for pair in date_ranges):
                raise ValueError("`date_ranges` must be a list of (datetime, datetime) tuples.")
            for start, end in date_ranges:
                if not isinstance(start, datetime) or not isinstance(end, datetime):
                    raise TypeError("Each element in `date_ranges` must be a (datetime, datetime) tuple.")
                if start >= end:
                    raise ValueError(f"Each `date_range` must be in the format (start < end), but got ({start}, {end}).")
            self.date_ranges = date_ranges

            sorted_ranges = sorted(self.date_ranges, key=lambda x: x[0])
            for i in range(len(sorted_ranges) - 1):
                if sorted_ranges[i][1] >= sorted_ranges[i+1][0]:
                    raise ValueError(f"`date_ranges` has overlapping periods: {sorted_ranges[i]} and {sorted_ranges[i+1]}")
            
            for start, end in self.date_ranges:
                if (end - start).days < 1:
                    raise ValueError(f"`date_ranges` segment ({start} to {end}) is too short or empty.")


        # ============================
        # VALIDATION: ref_date
        # ============================
        if ref_date is not None and not isinstance(ref_date, datetime):
            raise TypeError("`ref_date` must be a datetime object. Example: datetime(2024, 12, 1)")

        # Default to last day of date_ranges
        self.ref_date = ref_date or max(end for _, end in self.date_ranges)

        # Must be within date_ranges
        range_start = min(start for start, _ in self.date_ranges)
        range_end = max(end for _, end in self.date_ranges)

        if not (range_start <= self.ref_date <= range_end):
            raise ValueError(
                f"`ref_date` ({self.ref_date.date()}) must be within the date_ranges "
                f"({range_start.date()} to {range_end.date()})."
            )

        # If ref_date + booking_horizon exceeds range_end, extend the last date_range
        ref_end = self.ref_date + timedelta(days=booking_horizon)
        if ref_end > range_end:
            last_start, _ = self.date_ranges[-1]
            self.date_ranges[-1] = (last_start, ref_end)


        # ============================
        # VALIDATION: working_days
        # ============================
        wdays = working_days or [0, 1, 2, 3, 4]
        if not isinstance(wdays, list) or not all(isinstance(d, int) and 0 <= d <= 6 for d in wdays):
            raise ValueError("`working_days` must be a list of integers from 0 (Monday) to 6 (Sunday). Example: [0,1,2,3,4]")
        if not wdays:
            raise ValueError("`working_days` must not be an empty list. At least one working day is required.")
        if len(set(wdays)) != len(wdays):
            raise ValueError("`working_days` contains duplicate entries.")
        self.working_days = wdays


        # ============================
        # VALIDATION: appointments_per_hour
        # ============================
        valid_values = [1, 2, 3, 4, 6]
        slot_lengths = {v: 60 // v for v in valid_values}

        if not isinstance(appointments_per_hour, int) or appointments_per_hour not in valid_values:
            allowed = ", ".join([f"{v} ({slot_lengths[v]} min)" for v in valid_values])
            raise ValueError(
                f"`appointments_per_hour` must be one of: {allowed}. "
                "Only values that divide 60 evenly are allowed."
            )
        self.appointments_per_hour = appointments_per_hour


        # ============================
        # VALIDATION: working_hours
        # ============================
        self.working_hours = working_hours or [(8, 18)]

        if not self.working_hours:
            raise ValueError("`working_hours` must not be empty.")
        
        for i, (start, end) in enumerate(self.working_hours):
            if not (0 <= start < end <= 24):
                raise ValueError(f"`working_hours[{i}]` has invalid time range: ({start}, {end}). Must be 0 ≤ start < end ≤ 24.")
        # Optional: check for overlaps
        for i in range(len(self.working_hours)):
            for j in range(i+1, len(self.working_hours)):
                if max(self.working_hours[i][0], self.working_hours[j][0]) < min(self.working_hours[i][1], self.working_hours[j][1]):
                    raise ValueError(f"`working_hours[{i}]` and `working_hours[{j}]` overlap. Time blocks must not overlap.")
                
        for i, (start, end) in enumerate(self.working_hours):
            if (end - start) * 60 < 60 / self.appointments_per_hour:
                raise ValueError(
                    f"`working_hours[{i}]` does not allow for any slots given `appointments_per_hour = {self.appointments_per_hour}`"
                )


        # ============================
        # VALIDATION: fill_rate
        # ============================
        if not isinstance(fill_rate, (float, int)) or not (0.2 <= fill_rate <= 1.0):
            raise ValueError("`fill_rate` must be a float between 0.2 and 1.0. Values below 0.2 imply implausibly low utilization.")
        self.fill_rate = float(fill_rate)


        # ============================
        # VALIDATION: booking_horizon
        # ============================
        if not isinstance(booking_horizon, int) or not (0 <= booking_horizon <= 150):
            raise ValueError("`booking_horizon` must be an integer between 0 and 150 (approx. 5 months).")
        self.booking_horizon = booking_horizon


        # ============================
        # VALIDATION: median_lead_time
        # ============================
        if not isinstance(median_lead_time, int) or not (1 <= median_lead_time <= self.booking_horizon):
            raise ValueError(
                "`median_lead_time` must be an integer ≥ 1 and ≤ `booking_horizon`. "
                f"Received median_lead_time={median_lead_time}, booking_horizon={self.booking_horizon}."
            )
        self.median_lead_time = median_lead_time

        if self.booking_horizon >= 10 and self.median_lead_time < 2:
            warnings.warn(
                "`median_lead_time` is very short compared to `booking_horizon`. "
                "This may produce unnatural scheduling intervals."
            )


        # ============================
        # VALIDATION: status_rates
        # ============================
        expected_keys = {"attended", "cancelled", "did not attend", "unknown"}

        if status_rates is None:
            self.status_rates = DEFAULT_STATUS_RATES.copy()
        else:
            if not isinstance(status_rates, dict):
                raise TypeError("`status_rates` must be a dictionary.")
            if set(status_rates.keys()) != expected_keys:
                raise ValueError(f"`status_rates` must contain exactly the following keys: {sorted(expected_keys)}.")
            if not all(isinstance(v, (int, float)) and v >= 0 for v in status_rates.values()):
                raise ValueError("All values in `status_rates` must be non-negative numbers.")
            if sum(status_rates.values()) == 0:
                raise ValueError("The sum of `status_rates` values must be greater than 0.")
            
            self.status_rates = status_rates.copy()

        # Normalize to ensure the sum equals 1
        total = sum(self.status_rates.values())
        self.status_rates = {k: v / total for k, v in self.status_rates.items()}


        # ============================
        # VALIDATION: rebook_category
        # ============================
        valid_rebook_options = {"min", "med", "max"}
        if rebook_category not in valid_rebook_options:
            raise ValueError("`rebook_category` must be one of: 'min', 'med', or 'max'.")

        self.rebook_category = rebook_category
        self.rebook_ratio = {"min": 0, "med": 0.5, "max": 1}[rebook_category]


        # ============================
        # VALIDATION: check_in_time_mean
        # ============================
        if not isinstance(check_in_time_mean, (int, float)) or not (-60 <= check_in_time_mean <= 30):
            raise ValueError(
                "`check_in_time_mean` must be a number between -60 and +30. "
                "Negative values mean patients typically arrive early."
            )
        self.check_in_time_mean = check_in_time_mean


        # ============================
        # VALIDATION: visits_per_year
        # ============================
        if not isinstance(visits_per_year, (int, float)) or not (0.5 <= visits_per_year <= 10):
            raise ValueError("`visits_per_year` must be a float between 0.5 and 10. Values outside this range are implausible for outpatient care.")
        self.visits_per_year = float(visits_per_year)


        # ============================
        # VALIDATION: first_attendance
        # ============================
        if first_attendance is not None:
            if not isinstance(first_attendance, (float, int)) or not (0 <= first_attendance <= 1):
                raise ValueError("`first_attendance` must be a float between 0 and 1, or None to use default.")
            self.first_attendance = float(first_attendance)
        else:
            self.first_attendance = DEFAULT_FIRST_ATTENDANCE_RATIO


        # ============================
        # VALIDATION: lower_cutoff and upper_cutoff
        # ============================
        if not isinstance(lower_cutoff, int) or not (0 <= lower_cutoff <= 100):
            raise ValueError("`lower_cutoff` must be an integer between 0 and 100.")
        if not isinstance(upper_cutoff, int) or not (lower_cutoff < upper_cutoff <= 120):
            raise ValueError("`upper_cutoff` must be an integer greater than `lower_cutoff`, and no more than 120.")
        self.lower_cutoff = lower_cutoff
        self.upper_cutoff = upper_cutoff

        # ============================
        # VALIDATION: bin_size
        # ============================
        if not isinstance(bin_size, int) or not (1 <= bin_size <= 20):
            raise ValueError("`bin_size` must be an integer between 1 and 20.")
        self.bin_size = bin_size

        # ============================
        # VALIDATION: truncated
        # ============================
        if not isinstance(truncated, bool):
            raise TypeError("`truncated` must be a boolean.")
        self.truncated = truncated

        # ============================
        # VALIDATION: age_gender_probs
        # ============================
        if age_gender_probs is not None:
            required_cols = {"age_yrs", "total_male", "total_female"}
            if not isinstance(age_gender_probs, pd.DataFrame):
                raise TypeError("`age_gender_probs` must be a pandas DataFrame or None.")
            if not required_cols.issubset(age_gender_probs.columns):
                raise ValueError("`age_gender_probs` must contain columns: 'age_yrs', 'total_male', and 'total_female'.")
            self.age_gender_probs = age_gender_probs.copy()
        else:
            self.age_gender_probs = pd.DataFrame(DEFAULT_AGE_GENDER_PROBS.copy())


        # ============================
        # VALIDATION: seed
        # ============================
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("`seed` must be a non-negative integer.")
        self.seed = seed


        # ============================
        # VALIDATION: noise
        # ============================
        if not isinstance(noise, (int, float)) or not (0 <= noise <= 0.5):
            raise ValueError("`noise` must be a float between 0 and 0.5. Larger values may produce unstable or unrealistic variability.")
        self.noise = float(noise)



        # Initialize RNG and Faker
        self.fake = Faker()
        np.random.seed(self.seed)
        Faker.seed(self.seed)

        # Internal tracking
        self.total_appointments = 0
        self.fill_rate_calculated = 0.0
        self.scheduling_interval_mean_calculated = 0.0
        self.patients_df = pd.DataFrame()
        self.patient_id_counter = 1

        self.total_days = sum((end - start).days + 1 for start, end in self.date_ranges)
        self.slots_per_day = sum(
            (end - start) * self.appointments_per_hour
            for start, end in self.working_hours
        )
        self.working_days_count = sum(1 for date_day in pd.date_range(self.date_ranges[0][0], self.date_ranges[-1][1]) if date_day.weekday() in self.working_days)
        self.slots_per_week = self.slots_per_day * len(self.working_days)



    # -------------------------------------------------------------
    # Generate all appointment slots based on working calendar
    # -------------------------------------------------------------
    def generate_slots(self) -> pd.DataFrame:
        """
        Generate all available appointment slots based on configured working hours, days, and date ranges.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per slot. Each slot corresponds to a specific time block on a working day.

        Columns
        -------
        - slot_id : str
            Unique identifier for the slot (zero-padded).
        - appointment_date : datetime
            Date of the appointment slot.
        - appointment_time : time
            Start time of the slot.
        - is_available : bool
            True if the slot is available for booking. All slots are initialized as available.
        """

        # Get all working dates across date_ranges
        all_working_dates = []
        for start_date, end_date in self.date_ranges:
            days = pd.date_range(start=start_date, end=end_date)
            working_days = days[days.weekday.isin(self.working_days)]
            all_working_dates.extend(working_days)

        # Build time blocks for a single working day
        slot_minutes = []
        for start_hour, end_hour in self.working_hours:
            for hour in range(start_hour, end_hour):
                for m in range(0, 60, 60 // self.appointments_per_hour):
                    slot_minutes.append(time(hour=hour, minute=m))

        # Cartesian product: dates × time slots
        all_combinations = list(product(all_working_dates, slot_minutes))
        total_slots = len(all_combinations)
        max_digits = len(str(total_slots)) + 1

        # Vectorized construction
        slots_df = pd.DataFrame(all_combinations, columns=["appointment_date", "appointment_time"])
        slots_df["slot_id"] = [str(i).zfill(max_digits) for i in range(1, total_slots + 1)]
        slots_df["is_available"] = True
        slots_df["appointment_date"] = pd.to_datetime(slots_df["appointment_date"])
        slots_df = slots_df[["slot_id", "appointment_date", "appointment_time", "is_available"]]

        return slots_df


    # ---------------------------------------------------------------
    # Generate appointment bookings with realistic temporal dynamics
    # ---------------------------------------------------------------
    def generate_appointments(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simulate past and future appointment scheduling using realistic attendance and booking behavior.

        This method:
        - Selects a subset of past available slots to be booked.
        - Assigns scheduling dates based on an exponential distribution (median lead time).
        - Applies cancellation and rebooking logic.
        - Simulates future appointments post reference date.
        - Updates slot availability and calculates fill rate.

        Returns
        -------
        tuple
            slots_df : pd.DataFrame
                Updated slot availability.
            appointments_df : pd.DataFrame
                Final list of appointments with derived fields.
        """
        appointments = []
        max_digits = len(str(self.total_slots)) + 1

        # ------------------------------------------
        # Estimate number of initial appointments A0
        # ------------------------------------------
        c = self.status_rates['cancelled']
        s_attended = self.status_rates['attended']
        s_dna = self.status_rates['did not attend']
        s_unknown = self.status_rates['unknown']
        p_scheduled = s_attended + s_dna + s_unknown
        p_attended_given_scheduled = s_attended / p_scheduled

        A0 = (self.fill_rate * self.total_slots_past * (1 - c * self.rebook_ratio)) / ((1 - c) * p_attended_given_scheduled)
        A0 = min(int(round(A0)), self.total_slots_past)

        # ------------------------------------------
        # Sample past slots and mark them as booked
        # ------------------------------------------
        available_past_slots = self.slots_df[
            (self.slots_df["appointment_date"] < self.ref_date) &
            (self.slots_df["is_available"])
        ]

        A0 = min(A0, len(available_past_slots))
        booked_slots = available_past_slots.sample(n=A0, random_state=self.seed)
        self.slots_df.loc[self.slots_df["slot_id"].isin(booked_slots["slot_id"]), "is_available"] = False

        # ------------------------------------------
        # Schedule each appointment with lead time
        # ------------------------------------------
        for _, slot in booked_slots.iterrows():
            appointment_date = slot["appointment_date"]
            max_sched_interval = min(
                (appointment_date - self.earliest_scheduling_date).days,
                self.booking_horizon
            )

            tau_eff = self.median_lead_time / np.log(2)
            k = np.arange(1, max_sched_interval + 1)
            p = np.exp(-k / tau_eff)
            p /= p.sum()

            interval = int(np.random.choice(k, p=p))
            scheduling_date = max(appointment_date - timedelta(days=interval), self.earliest_scheduling_date)

            appointments.append({
                "slot_id": slot["slot_id"],
                "scheduling_date": scheduling_date,
                "appointment_date": appointment_date,
                "appointment_time": slot["appointment_time"],
                "scheduling_interval": (appointment_date - scheduling_date).days,
                "rebook_iteration": 0
            })

        self.appointments_df = pd.DataFrame(appointments)

        # ------------------------------------------
        # Apply cancellation + rebooking logic
        # ------------------------------------------
        self.appointments_df["primary_status"] = np.random.choice(
            ["cancelled", "scheduled"],
            size=len(self.appointments_df),
            p=[c, 1 - c]
        )

        self.appointments_df, appointment_id_counter = self.rebook_appointments(self.appointments_df)
        self.appointments_df = self.assign_status(self.appointments_df)

        # ------------------------------------------
        # Update slot availability based on outcome
        # ------------------------------------------
        cancelled_ids = self.appointments_df[self.appointments_df["primary_status"] == "cancelled"]["slot_id"]
        attended_ids = self.appointments_df[
            self.appointments_df["status"].isin(["attended", "did not attend", "unknown"])
        ]["slot_id"]

        self.slots_df.loc[self.slots_df["slot_id"].isin(cancelled_ids), "is_available"] = True
        self.slots_df.loc[self.slots_df["slot_id"].isin(attended_ids), "is_available"] = False

        self.fill_rate_calculated = (
            len(attended_ids) / self.total_slots_past
            if self.total_slots_past > 0 else 0
        )

        # ------------------------------------------
        # Generate future appointments post ref_date
        # ------------------------------------------
        self.appointments_df = self.schedule_future_appointments(self.appointments_df, appointment_id_counter)
        self.appointments_df = self.appointments_df.sort_values(by="scheduling_date").reset_index(drop=True)

        max_digits = len(str(len(self.appointments_df))) + 1
        self.appointments_df["appointment_id"] = self.appointments_df.index + 1
        self.appointments_df["appointment_id"] = self.appointments_df["appointment_id"].astype(str).str.zfill(max_digits)

        # ------------------------------------------
        # Finalize structure and add real-time fields
        # ------------------------------------------
        self.appointments_df = self.appointments_df.drop(columns=["rebook_iteration", "primary_status"])
        self.appointments_df = self.appointments_df[[
            "appointment_id", "slot_id", "scheduling_date", "appointment_date",
            "appointment_time", "scheduling_interval", "status"
        ]]

        self.appointments_df = self.appointments_df.sort_values(by=["appointment_date", "appointment_time"]).reset_index(drop=True)
        self.appointments_df = self.assign_actual_times(self.appointments_df)

        # ------------------------------------------
        # Track average scheduling delay
        # ------------------------------------------
        if len(self.appointments_df) > 0:
            self.scheduling_interval_mean_calculated = self.appointments_df["scheduling_interval"].mean()

        return self.slots_df, self.appointments_df


    # ---------------------------------------------------------------
    # Rebooking Logic for Cancelled Appointments (Iterative Simulation)
    # ---------------------------------------------------------------
    def rebook_appointments(self, appointments_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Rebook a fraction of cancelled appointments according to the rebooking policy.

        This method simulates the real-world behavior where some cancelled appointments
        are rescheduled. The process is controlled by:
        - `rebook_ratio`: Proportion of cancelled appointments that are rebooked.
        - `rebook_category`: Limits the number of iterations ('min', 'med', 'max').

        Parameters
        ----------
        appointments_df : pd.DataFrame
            DataFrame containing initial appointment records with `primary_status` and `rebook_iteration`.

        Returns
        -------
        tuple
            updated_df : pd.DataFrame
                Appointments including all rebooked entries.
            appointment_id_counter : int
                Last used appointment ID counter.
        """
        appointment_id_counter = 1
        max_digits = len(str(self.total_slots)) + 1
        rebook_iteration = 0

        while True:
            # -------------------------------
            # Select cancelled appointments
            # -------------------------------
            cancelled = appointments_df[
                (appointments_df["primary_status"] == "cancelled") &
                (appointments_df["rebook_iteration"] == rebook_iteration)
            ]

            if cancelled.empty or len(cancelled) <= 1:
                break

            rebook_count = int(len(cancelled) * self.rebook_ratio)
            if rebook_count == 0:
                break

            rebooked = cancelled.sample(n=rebook_count, random_state=self.seed + rebook_iteration)
            new_appointments = []

            # -------------------------------
            # Schedule rebooked appointments
            # -------------------------------
            for _, original_appointment in rebooked.iterrows():
                days_between = (original_appointment["appointment_date"] - original_appointment["scheduling_date"]).days

                if days_between <= 1:
                    continue  # Not enough room to rebook

                days_after = np.random.randint(1, days_between)
                new_sched_date = original_appointment["scheduling_date"] + timedelta(days=days_after)

                if new_sched_date >= original_appointment["appointment_date"]:
                    continue  # Invalid rebooking date

                new_appt = {
                    "appointment_id": str(appointment_id_counter).zfill(max_digits),
                    "slot_id": original_appointment["slot_id"],
                    "scheduling_date": new_sched_date,
                    "appointment_date": original_appointment["appointment_date"],
                    "appointment_time": original_appointment["appointment_time"],
                    "scheduling_interval": (original_appointment["appointment_date"] - new_sched_date).days,
                    "rebook_iteration": rebook_iteration + 1,
                    "primary_status": np.random.choice(
                        ["cancelled", "scheduled"],
                        p=[self.status_rates["cancelled"], 1 - self.status_rates["cancelled"]]
                    )
                }

                new_appointments.append(new_appt)
                appointment_id_counter += 1

            # Add new rebooked appointments
            if new_appointments:
                appointments_df = pd.concat(
                    [appointments_df, pd.DataFrame(new_appointments)],
                    ignore_index=True
                )
            else:
                break

            rebook_iteration += 1

            # -------------------------------
            # Control iteration depth
            # -------------------------------
            if self.rebook_category == "min":
                break  # Always stop after 1
            elif self.rebook_category == "med" and rebook_iteration >= 2:
                break
            elif self.rebook_category == "max":
                remaining = appointments_df[
                    (appointments_df["primary_status"] == "cancelled") &
                    (appointments_df["rebook_iteration"] == rebook_iteration)
                ]
                if len(remaining) <= 1:
                    break

        return appointments_df, appointment_id_counter


    # ---------------------------------------------------------------
    # Assign Final Status to Appointments Before Reference Date
    # ---------------------------------------------------------------
    def assign_status(self, appointments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign the final 'status' to each appointment based on its primary state
        and the reference date.

        Appointments scheduled before the reference date and not cancelled are randomly
        assigned one of: 'attended', 'did not attend', or 'unknown', based on normalized
        probabilities from the configured status rates.

        Cancelled appointments are directly labeled with 'status' = 'cancelled'.

        Parameters
        ----------
        appointments_df : pd.DataFrame
            Appointments with columns: 'appointment_date', 'primary_status'.

        Returns
        -------
        pd.DataFrame
            Updated DataFrame with a new 'status' column.
        """
        # ---------------------------------------------------------
        # Handle pre-ref_date appointments that were not cancelled
        # ---------------------------------------------------------
        mask = (
            (appointments_df["appointment_date"] < self.ref_date) &
            (appointments_df["primary_status"] != "cancelled")
        )
        pre_ref_active = appointments_df[mask]

        if len(pre_ref_active) > 0:
            s_att = self.status_rates["attended"]
            s_dna = self.status_rates["did not attend"]
            s_unk = self.status_rates["unknown"]
            total = s_att + s_dna + s_unk

            normalized_probs = {
                "attended": s_att / total,
                "did not attend": s_dna / total,
                "unknown": s_unk / total
            }

            status_labels = list(normalized_probs.keys())
            probabilities = list(normalized_probs.values())

            random_labels = np.random.choice(
                status_labels,
                size=len(pre_ref_active),
                p=probabilities
            )

            appointments_df.loc[pre_ref_active.index, "status"] = random_labels

        # ---------------------------------------------------------
        # Cancelled appointments retain their 'cancelled' status
        # ---------------------------------------------------------
        appointments_df.loc[
            appointments_df["primary_status"] == "cancelled",
            "status"
        ] = "cancelled"

        return appointments_df


    # ---------------------------------------------------------------
    # Simulate Realistic Arrival, Start, and End Times (Attended Only)
    # ---------------------------------------------------------------
    def assign_actual_times(self, appointments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate real-world appointment execution times for 'attended' visits.

        For each attended appointment:
        - Assign a check-in time based on punctuality behavior.
        - Simulate actual start time considering early arrivals, clinic backlog, and working hours.
        - Draw duration from a Beta distribution to reflect realistic variability.
        - Compute end time and patient waiting time.

        Parameters
        ----------
        appointments_df : pd.DataFrame
            DataFrame of all appointments, with at least:
            ['appointment_date', 'appointment_time', 'status'].

        Returns
        -------
        pd.DataFrame
            Updated DataFrame with timing fields:
            ['check_in_time', 'start_time', 'end_time', 'appointment_duration', 'waiting_time'].
        """
        # ---------------------------------------------------------------
        # Initialize columns
        # ---------------------------------------------------------------
        appointments_df["check_in_time"] = pd.NaT
        appointments_df["appointment_duration"] = np.nan
        appointments_df["start_time"] = pd.NaT
        appointments_df["end_time"] = pd.NaT
        appointments_df["waiting_time"] = np.nan

        attended = appointments_df[appointments_df["status"] == "attended"]
        slot_duration = 60 / self.appointments_per_hour  # in minutes

        # ---------------------------------------------------------------
        # Simulate daily attended appointments
        # ---------------------------------------------------------------
        for date in attended["appointment_date"].unique():
            daily = attended[attended["appointment_date"] == date].copy()

            # Generate check-in times (some early, some late)
            daily["check_in_time"] = daily.apply(
                lambda row: self.generate_check_in_time(row["appointment_date"], row["appointment_time"]),
                axis=1
            )

            # Order patients by arrival time (to simulate real queueing)
            daily.sort_values(by=["check_in_time", "appointment_time"], inplace=True)

            previous_end = None

            for idx, appt in daily.iterrows():
                sched_dt = datetime.combine(appt["appointment_date"], appt["appointment_time"])
                work_start = sched_dt.replace(hour=self.working_hours[0][0], minute=0, second=0)

                check_in = appt["check_in_time"]
                earliest_start = max(check_in, work_start)

                if previous_end:
                    earliest_start = max(earliest_start, previous_end)

                # Simulate random service delay (normal dist, ≥0)
                delay_sec = max(0, np.random.normal(loc=60, scale=(75 - 60) / 1.96))
                start = earliest_start + timedelta(seconds=delay_sec)

                # Draw duration from Beta(1.48, 3.6) scaled to [0, 60]
                duration = round(np.random.beta(1.48, 3.6) * 60, 1)
                end = start + timedelta(minutes=duration)
                wait = round((start - check_in).total_seconds() / 60.0, 1)

                previous_end = end

                # Assign to main DataFrame
                appointments_df.at[appt.name, "check_in_time"] = check_in
                appointments_df.at[appt.name, "start_time"] = start
                appointments_df.at[appt.name, "end_time"] = end
                appointments_df.at[appt.name, "appointment_duration"] = duration
                appointments_df.at[appt.name, "waiting_time"] = wait

            # Update all new values into attended subset
            attended.update(daily)

        # ---------------------------------------------------------------
        # Format times to HH:MM:SS strings
        # ---------------------------------------------------------------
        for col in ["start_time", "end_time", "check_in_time"]:
            appointments_df[col] = pd.to_datetime(appointments_df[col]).dt.strftime("%H:%M:%S")

        return appointments_df


    # ---------------------------------------------------------------
    # Simulate Patient Check-In Time Based on Punctuality Behavior
    # ---------------------------------------------------------------
    def generate_check_in_time(self, appointment_date: date, appointment_time: time) -> datetime:
        """
        Simulate the patient’s check-in time relative to their scheduled appointment.

        This method models punctuality based on outpatient clinic observations:
        - ~84.4% of patients arrive early (Cerruti et al., 2023).
        - The average arrival time is 10 minutes before the scheduled slot (mode ≈ -10).
        - A normal distribution is used for variability around the mean.

        Returns
        -------
        datetime
            Simulated check-in timestamp (may be before or after the scheduled time).
        """
        scheduled_dt = datetime.combine(appointment_date, appointment_time)

        # Although punctuality is not strictly normally distributed (Cerruti et al., 2023),
        # we approximate it using a normal distribution centered on `check_in_time_mean`
        # with a fixed standard deviation of 9.8 minutes.
        # This ensures ~95% of arrivals fall within ±20 minutes around the mean,
        # producing plausible and stable arrival behavior across configurations.

        mean_offset_min = self.check_in_time_mean
        std_offset_min = 9.8  # Fixed std for ~95% within ±20 min

        offset = np.random.normal(loc=mean_offset_min, scale=std_offset_min)
        check_in_dt = scheduled_dt + timedelta(minutes=offset)

        return check_in_dt


    # ---------------------------------------------------------------
    # Simulate Future Appointments with Decaying Fill and Cancellation Rates
    # ---------------------------------------------------------------
    def schedule_future_appointments(self, appointments_df: pd.DataFrame, appointment_id_counter: int) -> pd.DataFrame:
        """
        Schedule appointments after the reference date using a decaying fill rate
        and a time-dependent cancellation probability.

        This simulates how clinics experience lower booking density and higher
        cancellation uncertainty as appointment dates move farther into the future.

        Parameters
        ----------
        appointments_df : pd.DataFrame
            Existing appointment DataFrame to which future appointments will be appended.
        appointment_id_counter : int
            Initial counter for assigning unique appointment IDs.

        Returns
        -------
        pd.DataFrame
            Updated appointment DataFrame including simulated future appointments.
        """
        # -------------------------------------------------------
        # Identify future appointment dates within booking horizon
        # -------------------------------------------------------
        slots_per_date = self.slots_df.groupby("appointment_date").size()
        future_dates = slots_per_date.index[
            (slots_per_date.index >= self.ref_date) &
            (slots_per_date.index < self.ref_date + timedelta(days=self.booking_horizon))
        ]

        # -------------------------------------------------------
        # Precompute decay constants
        # -------------------------------------------------------
        tau_eff = self.median_lead_time / np.log(2)
        k_fill = -np.log(0.01 / self.fill_rate) / self.booking_horizon
        k_cancel = -np.log(0.01 / self.status_rates["cancelled"]) / self.booking_horizon

        for appt_date in future_dates:
            delta = (appt_date - self.ref_date).days

            # Decaying fill and cancel rates
            fill_rate = self.fill_rate * np.exp(-k_fill * delta)
            cancel_rate = self.status_rates["cancelled"] * np.exp(-k_cancel * delta)

            variability = np.random.normal(loc=1, scale=self.noise)
            expected_n = int(slots_per_date[appt_date] * fill_rate * variability)
            expected_n = max(0, min(expected_n, slots_per_date[appt_date]))

            available = self.slots_df[
                (self.slots_df["appointment_date"] == appt_date) &
                (self.slots_df["is_available"])
            ]

            if expected_n > len(available):
                expected_n = len(available)

            if expected_n > 0:
                booked = available.sample(n=expected_n, random_state=self.seed + delta)
                self.slots_df.loc[self.slots_df["slot_id"].isin(booked["slot_id"]), "is_available"] = False

                new_appointments = []
                for _, slot in booked.iterrows():
                    # Schedule date based on exponential lead time distribution
                    max_interval = min(delta, self.booking_horizon)
                    sched_interval = int(round(np.random.exponential(scale=tau_eff)))
                    sched_interval = max(1, min(sched_interval, max_interval))

                    sched_date = appt_date - timedelta(days=sched_interval)
                    sched_date = min(sched_date, self.ref_date)
                    sched_date = max(sched_date, self.ref_date - timedelta(days=self.booking_horizon))

                    appt = {
                        "appointment_id": str(appointment_id_counter).zfill(len(str(self.total_slots)) + 1),
                        "slot_id": slot["slot_id"],
                        "scheduling_date": sched_date,
                        "appointment_date": appt_date,
                        "appointment_time": slot["appointment_time"],
                        "scheduling_interval": (appt_date - sched_date).days,
                        "status": "scheduled"
                    }

                    # Assign cancellation based on decaying probability
                    appt["status"] = np.random.choice(
                        ["cancelled", "scheduled"],
                        p=[cancel_rate, 1 - cancel_rate]
                    )

                    if appt["status"] == "cancelled":
                        self.slots_df.loc[self.slots_df["slot_id"] == slot["slot_id"], "is_available"] = True

                    new_appointments.append(appt)
                    appointment_id_counter += 1

                appointments_df = pd.concat([appointments_df, pd.DataFrame(new_appointments)], ignore_index=True)

        return appointments_df

    # ---------------------------------------------------------------
    # Generate Synthetic Patient Demographics by Age and Sex Distribution
    # ---------------------------------------------------------------
    def generate_patients(self, total_patients: int) -> pd.DataFrame:
        """
        Generate a synthetic patient population using demographic probabilities
        from NHS England (2023–24), stratified by age and sex.

        Patients are sampled proportionally from a predefined age-sex distribution.
        Age ranges are drawn from demographic bins, then ages are randomized within those bins.

        Parameters
        ----------
        total_patients : int
            Total number of unique patients to generate.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns: patient_id, name, sex, age.
        """
        # Parse age range bins
        age_ranges = [
            (90, 100) if age == "90+" else tuple(map(int, age.split('-')))
            for age in self.age_gender_probs["age_yrs"]
        ]

        # Apply truncation filter if enabled
        if self.truncated:
            full_ranges = age_ranges
            valid_idx = [i for i, (low, high) in enumerate(full_ranges) if high >= self.lower_cutoff]
            self.age_gender_probs = self.age_gender_probs.iloc[valid_idx].reset_index(drop=True)
            age_ranges = [full_ranges[i] for i in valid_idx]

        # Compute proportional sex ratios
        female_prop = self.age_gender_probs["total_female"].sum()
        male_prop = self.age_gender_probs["total_male"].sum()
        total_prop = female_prop + male_prop
        num_females = int(total_patients * (female_prop / total_prop))
        num_males = total_patients - num_females

        # Generate synthetic patients
        patients = []

        # Female sampling
        female_probs = self.age_gender_probs["total_female"] / female_prop
        for i in np.random.choice(len(age_ranges), size=num_females, p=female_probs):
            low, high = age_ranges[i]
            age = np.random.randint(max(low, self.lower_cutoff), min(high, self.upper_cutoff) + 1)
            name = self.fake.name_female()
            patients.append({"name": name, "sex": "Female", "age": age})

        # Male sampling
        male_probs = self.age_gender_probs["total_male"] / male_prop
        for i in np.random.choice(len(age_ranges), size=num_males, p=male_probs):
            low, high = age_ranges[i]
            age = np.random.randint(max(low, self.lower_cutoff), min(high, self.upper_cutoff) + 1)
            name = self.fake.name_male()
            patients.append({"name": name, "sex": "Male", "age": age})

        # Create patient_id and assemble DataFrame
        patients_df = pd.DataFrame(patients)
        id_length = max(5, len(str(self.patient_id_counter + total_patients - 1)))
        ids = [f"{i:0{id_length}d}" for i in range(self.patient_id_counter, self.patient_id_counter + total_patients)]
        self.patient_id_counter += total_patients
        patients_df.insert(0, "patient_id", ids)

        return patients_df


    # ------------------------------------------------------------------------------
    # Assign Patients to Appointments with Controlled Turnover and Demographic Logic
    # ------------------------------------------------------------------------------
    def assign_patients(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Assigns synthetic patients to scheduled appointments across simulation years.

        The method simulates annual patient turnover using a configurable
        `first_attendance` rate. It generates new patients and reuses existing ones,
        assigning appointments based on expected visit frequency and demographic distributions.

        It also computes each patient’s date of birth (dob) based on their age and
        first appointment, and dynamically assigns age and age_group to each appointment.

        Returns:
            tuple:
                - patients_df (pd.DataFrame): Final patient registry with `patient_id`, `name`, `sex`, `dob`, and `insurance`.
                - appointments_df (pd.DataFrame): Updated appointments with `patient_id`, `age`, and `age_group`.
        """
        # Sort appointments by date and extract year
        self.appointments_df = self.appointments_df.sort_values('appointment_date').reset_index(drop=True)
        self.appointments_df['appointment_year'] = self.appointments_df['appointment_date'].dt.year

        # Compute yearly appointment volumes
        appointments_per_year = self.appointments_df.groupby('appointment_year').size().to_dict()

        # Initialize accumulators
        patients_df = pd.DataFrame()
        assigned_appointments_df = pd.DataFrame()
        accumulated_patients = pd.DataFrame()

        # Process appointments by year
        for idx, (year, total_appointments) in enumerate(appointments_per_year.items()):
            yearly_random_factor = 1 + np.random.uniform(-self.noise, self.noise)

            if idx == 0:
                # First year: generate all new patients
                num_patients = max(1, int((total_appointments / self.visits_per_year) * yearly_random_factor))
                new_patients_df = self.generate_patients(num_patients)
                new_patients_df['year_joined'] = year

                accumulated_patients = new_patients_df.copy()
                patients_df = new_patients_df.copy()
                current_patients_df = new_patients_df.copy()
            else:
                # Subsequent years: mix of new and returning patients
                expected_total_patients = max(1, int((total_appointments / self.visits_per_year) * yearly_random_factor))

                new_patient_ratio = self.first_attendance * (1 + np.random.uniform(-self.noise, self.noise))
                num_new_patients = max(0, int(expected_total_patients * new_patient_ratio))
                num_returning_patients = expected_total_patients - num_new_patients

                # Limit to available pool
                num_returning_patients = min(num_returning_patients, len(accumulated_patients))
                num_new_patients = expected_total_patients - num_returning_patients

                # Sample returning patients and increment their age
                returning_df = accumulated_patients.sample(n=num_returning_patients, random_state=self.seed).copy()
                returning_df['age'] += 1

                # Generate new patients
                new_patients_df = self.generate_patients(num_new_patients)
                new_patients_df['year_joined'] = year

                # Optional age noise adjustment
                if num_new_patients > 0:
                    age_shift = num_returning_patients / num_new_patients
                    adjusted_ages = []
                    for age in new_patients_df['age']:
                        noise = np.random.uniform(-self.noise, self.noise)
                        adjusted_age = age - int(age_shift + noise)
                        while adjusted_age < self.lower_cutoff or adjusted_age > self.upper_cutoff:
                            adjusted_age = np.random.randint(self.lower_cutoff, self.upper_cutoff + 1)
                        adjusted_ages.append(adjusted_age)
                    new_patients_df['age'] = adjusted_ages

                # Update accumulators
                accumulated_patients = pd.concat([accumulated_patients, new_patients_df], ignore_index=True)
                accumulated_patients.drop_duplicates(subset='patient_id', inplace=True)
                patients_df = pd.concat([patients_df, new_patients_df], ignore_index=True)
                current_patients_df = pd.concat([new_patients_df, returning_df], ignore_index=True)

            # Assign appointments for this year
            year_appointments = self.appointments_df[self.appointments_df['appointment_year'] == year]
            num_appointments = len(year_appointments)

            # Visit distribution: ensure 1+ visit per patient
            visit_counts = np.ones(len(current_patients_df), dtype=int)
            remaining = num_appointments - len(current_patients_df)
            if remaining > 0:
                additional = np.random.choice(len(current_patients_df), size=remaining, replace=True)
                np.add.at(visit_counts, additional, 1)

            # Build patient_id list with repetitions
            patient_ids = []
            for pid, count in zip(current_patients_df['patient_id'], visit_counts):
                patient_ids.extend([pid] * count)

            np.random.shuffle(patient_ids)
            year_appointments = year_appointments.copy()
            year_appointments['patient_id'] = patient_ids[:num_appointments]

            assigned_appointments_df = pd.concat([assigned_appointments_df, year_appointments], ignore_index=True)

        # Final patient registry
        self.patients_df = patients_df.drop_duplicates('patient_id').reset_index(drop=True)
        self.appointments_df = assigned_appointments_df.merge(self.patients_df, on='patient_id', how='left')

        # Calculate DOB from age and first appointment date
        first_appt = self.appointments_df.groupby('patient_id')['appointment_date'].min().reset_index()
        first_appt.rename(columns={'appointment_date': 'first_appointment_date'}, inplace=True)
        self.patients_df = self.patients_df.merge(first_appt, on='patient_id', how='left')

        self.patients_df['dob'] = self.patients_df.apply(
            lambda row: row['first_appointment_date'] - pd.DateOffset(years=row['age']) - pd.Timedelta(days=np.random.randint(0, 364)),
            axis=1
        )
        self.patients_df['dob'] = pd.to_datetime(self.patients_df['dob'])
        self.patients_df.drop(columns=['first_appointment_date', 'age', 'year_joined'], inplace=True)

        # Merge dob back into appointments and compute age
        self.appointments_df = self.appointments_df.merge(self.patients_df[['patient_id', 'dob']], on='patient_id', how='left')
        self.appointments_df['age'] = self.appointments_df.apply(
            lambda row: row['appointment_date'].year - row['dob'].year -
                        ((row['appointment_date'].month, row['appointment_date'].day) <
                        (row['dob'].month, row['dob'].day)),
            axis=1
        )

        # Age group binning
        lower = max(self.lower_cutoff, 15) if self.truncated else self.lower_cutoff
        bins = list(range(lower, self.upper_cutoff + 1, self.bin_size)) + [101]
        if self.upper_cutoff < 101:
            labels = [f'{bins[i]}-{bins[i+1]-1}' for i in range(len(bins)-2)] + [f'{self.upper_cutoff}+']
        else:
            labels = [f'{bins[i]}-{bins[i+1]-1}' for i in range(len(bins)-1)]

        self.appointments_df['age_group'] = pd.cut(self.appointments_df['age'], bins=bins, labels=labels, right=False)

        # Final cleanup
        self.appointments_df.drop(columns=['name', 'appointment_year', 'year_joined', 'dob'], inplace=True)
        return self.patients_df, self.appointments_df


    # ------------------------------------------------------------------
    # Internal methods for categorical distribution generation
    # ------------------------------------------------------------------

    def _pareto_distribution(self, categories):
        """
        Generate a Pareto-like (power-law) probability distribution over a list of categories.

        Args:
            categories (list): A list of category labels (e.g., insurance providers).

        Returns:
            np.ndarray: Normalized probabilities that follow a decreasing Pareto-like pattern with noise.
        """
        base_probs = np.array([1 / (i + 1) for i in range(len(categories))])
        base_probs /= base_probs.sum()

        # Apply random noise to simulate real-world irregularities
        noise_adjustment = np.random.uniform(1 - self.noise, 1 + self.noise, size=len(categories))
        adjusted_probs = base_probs * noise_adjustment
        return adjusted_probs / adjusted_probs.sum()


    def _uniform_distribution(self, categories):
        """
        Generate a uniform probability distribution with small randomness.

        Args:
            categories (list): A list of category labels.

        Returns:
            np.ndarray: Nearly uniform distribution with small variability introduced via noise.
        """
        base_probs = np.ones(len(categories)) / len(categories)
        noise_adjustment = np.random.uniform(1 - self.noise, 1 + self.noise, size=len(categories))
        adjusted_probs = base_probs * noise_adjustment
        return adjusted_probs / adjusted_probs.sum()


    def _normal_distribution(self, categories):
        """
        Generate a bell-shaped (normal-like) distribution centered on the middle category.

        Args:
            categories (list): A list of category labels.

        Returns:
            np.ndarray: Normalized distribution resembling a Gaussian shape, with noise applied.
        """
        n = len(categories)
        mean_idx = n / 2
        std_dev = n / 4
        indices = np.arange(n)

        base_probs = np.exp(-0.5 * ((indices - mean_idx) / std_dev) ** 2)
        base_probs /= base_probs.sum()

        noise_adjustment = np.random.uniform(1 - self.noise, 1 + self.noise, size=n)
        adjusted_probs = base_probs * noise_adjustment
        return adjusted_probs / adjusted_probs.sum()

    # ------------------------------------------------------------------
    # Add custom categorical column to patients_df
    # ------------------------------------------------------------------

    def add_custom_column(self, column_name, categories, distribution_type='normal', custom_probs=None):
        """
        Add a synthetic categorical column to the patients DataFrame using a specified probability distribution.

        Args:
            column_name (str): Name of the new column to be added (e.g., "insurance").
            categories (list): List of categorical labels to assign (e.g., insurance providers).
            distribution_type (str): Distribution type to use ("pareto", "uniform", or "normal").
            custom_probs (list, optional): Custom list of probabilities matching the order of `categories`.

        Raises:
            ValueError: If custom_probs length mismatches categories, or probabilities do not sum to 1.
        """
        if custom_probs is not None:
            if len(custom_probs) != len(categories):
                raise ValueError("Length of custom_probs must match the length of categories.")
            if not np.isclose(sum(custom_probs), 1.0):
                raise ValueError("custom_probs must sum to 1.")
            probs = custom_probs
        else:
            if distribution_type == 'pareto':
                probs = self._pareto_distribution(categories)
            elif distribution_type == 'uniform':
                probs = self._uniform_distribution(categories)
            elif distribution_type == 'normal':
                probs = self._normal_distribution(categories)
            else:
                raise ValueError("Invalid distribution_type. Choose from 'pareto', 'uniform', or 'normal'.")

        self.patients_df[column_name] = np.random.choice(categories, size=len(self.patients_df), p=probs)

    # ------------------------------------------------------------------------
    # Main pipeline method to generate slots, appointments, and patients
    # ------------------------------------------------------------------------
    def generate(self):
        """
        Execute the full generation pipeline:
        - Create available appointment slots
        - Simulate appointment scheduling and statuses
        - Assign synthetic patients and link them to appointments

        Returns:
            tuple:
                - slots_df: DataFrame of available appointment slots
                - appointments_df: DataFrame of scheduled appointments
                - patients_df: DataFrame of generated patients
        """
        # Generate appointment slots
        self.slots_df = self.generate_slots()
        self.total_slots = len(self.slots_df)
        self.total_slots_past = len(self.slots_df[self.slots_df['appointment_date'] < self.ref_date])

        # Compute scheduling bounds
        self.earliest_appointment_date = self.slots_df['appointment_date'].min().normalize()
        self.latest_appointment_date = self.slots_df['appointment_date'].max().normalize()
        self.earliest_scheduling_date = self.earliest_appointment_date - timedelta(days=self.booking_horizon)
        self.slots_per_date = self.slots_df.groupby('appointment_date').size()

        # Simulate appointments
        self.generate_appointments()

        # Estimate simulation period and required patient volume
        self.date_range_days = (self.appointments_df['appointment_date'].max() -
                                self.appointments_df['appointment_date'].min()).days
        self.total_years = max(int(np.ceil(self.date_range_days / 365.25)), 1)
        self.total_appointments = len(self.appointments_df)
        self.expected_visits_per_patient = self.visits_per_year * self.total_years

        # Assign patients to appointments
        self.assign_patients()

        return self.slots_df, self.appointments_df, self.patients_df