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
# It also includes:
#   - NHS outpatient data parsing for setting real-world default parameters
#   - Utility methods for demographic and categorical data assignment
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

# Core Libraries
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta, date, time
from typing import List, Tuple, Optional

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Synthetic Data
from faker import Faker

# ---------------------------------------------------------------------------------------------------
# NHS Outpatient Data Parser (Optional Reference)
# ---------------------------------------------------------------------------------------------------
# This utility function loads and processes outpatient statistics from NHS England (2023–24).
# It was used to derive the default parameters for:
# - DEFAULT_AGE_GENDER_PROBS         ← Summary Report 3
# - DEFAULT_FIRST_ATTENDANCE_RATIO   ← Summary Report 2
# - DEFAULT_STATUS_RATES             ← Summary Report 1
#
# The function is included for transparency and reproducibility but is NOT required at runtime.
# It can be used manually to refresh the constant values in this module.
# ---------------------------------------------------------------------------------------------------

def parse_nhs_outpatient_data(url: str) -> tuple[pd.DataFrame, float, dict]:
    """
    Parses NHS England outpatient data from the official Excel workbook and returns:

    - A DataFrame with normalized female and male attendance percentages by age group.
    - The national first attendance ratio (first / total attendances).
    - A dictionary of appointment status proportions for 2023–24.

    Parameters
    ----------
    url : str
        Direct URL to the NHS Excel file (e.g., from https://digital.nhs.uk).

    Returns
    -------
    tuple
        DEFAULT_AGE_GENDER_PROBS : pd.DataFrame
            Columns: ['age_yrs', 'total_female', 'total_male'], values normalized to sum ≈ 1.
        DEFAULT_FIRST_ATTENDANCE_RATIO : float
            First attendances / total attendances (rounded to 5 decimals).
        DEFAULT_STATUS_RATES : dict
            Keys: {'attended', 'cancelled', 'did not attend', 'unknown'} with values ∈ [0, 1].
    """
    try:
        summary3 = pd.read_excel(url, sheet_name="Summary Report 3", header=5, nrows=20)
        summary2 = pd.read_excel(url, sheet_name="Summary Report 2", header=5, nrows=9)
        summary1 = pd.read_excel(url, sheet_name="Summary Report 1", header=5, nrows=12)
    except Exception as e:
        print(f"Error loading data from URL: {e}")
        return pd.DataFrame(), None, {}

    # --- Age and sex distribution (Summary Report 3) ---
    summary3.columns = [
        'age_yrs', 'attended_female_maternity', 'attended_female_non_maternity',
        'attended_male', 'dna_female', 'dna_male'
    ]
    summary3['attended_female'] = (
        summary3['attended_female_maternity'] + summary3['attended_female_non_maternity']
    )

    relevant_cols = ['attended_female', 'attended_male', 'dna_female', 'dna_male']
    total_count = summary3[relevant_cols].to_numpy().sum()

    if total_count == 0:
        return pd.DataFrame(), None, {}

    summary3[relevant_cols] /= total_count
    summary3['total_female'] = summary3['attended_female'] + summary3['dna_female']
    summary3['total_male'] = summary3['attended_male'] + summary3['dna_male']
    DEFAULT_AGE_GENDER_PROBS = summary3[['age_yrs', 'total_female', 'total_male']].round(5)

    # --- First attendance ratio (Summary Report 2) ---
    try:
        row = summary2[summary2.iloc[:, 0] == 'Total Activity'].squeeze()
        DEFAULT_FIRST_ATTENDANCE_RATIO = round(
            row['First Attendances'] / row['Attendances'], 5
        )
    except Exception as e:
        print(f"Error extracting first attendance ratio: {e}")
        DEFAULT_FIRST_ATTENDANCE_RATIO = None

    # --- Status distribution for 2023–24 (Summary Report 1) ---
    try:
        row = summary1[summary1['Year'] == '2023-24'].squeeze()
        DEFAULT_STATUS_RATES = {
            "attended": round(row['Attendances %'] / 100, 3),
            "did not attend": round(row['Did not attends (DNAs) %'] / 100, 3),
            "cancelled": round(
                (row['Patient cancellations %'] + row['Hospital cancellations %']) / 100, 3
            ),
            "unknown": round(row['Unknown %'] / 100, 3)
        }
    except Exception as e:
        print(f"Error extracting status rates: {e}")
        DEFAULT_STATUS_RATES = {}

    return DEFAULT_AGE_GENDER_PROBS, DEFAULT_FIRST_ATTENDANCE_RATIO, DEFAULT_STATUS_RATES


# -----------------------------------------------------------------------------
# Optional: To regenerate constants manually, uncomment and run the block below
# -----------------------------------------------------------------------------
# nhs_url = "https://files.digital.nhs.uk/34/18846B/hosp-epis-stat-outp-rep-tabs-2023-24-tab.xlsx"
# DEFAULT_AGE_GENDER_PROBS, DEFAULT_FIRST_ATTENDANCE_RATIO, DEFAULT_STATUS_RATES = parse_nhs_outpatient_data(nhs_url)


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
        Default: [(2015-01-01, 2024-12-31)]

    working_days : list of int, optional
        Weekdays when the clinic operates (0=Monday, ..., 6=Sunday).
        Default: [0, 1, 2, 3, 4]

    working_hours : list of (int, int), optional
        Time blocks of daily working hours (e.g., [(8, 18)] for 8:00 to 18:00).
        Default: [(8, 18)]

    appointments_per_hour : int
        Number of bookable appointments per hour (defines slot duration).
        Default: 4 (i.e., 15-minute slots)

    fill_rate : float
        Target proportion of available slots that are eventually booked and attended.
        Default: 0.9

    booking_horizon : int
        Maximum number of days into the future that appointments can be scheduled.
        Default: 30

    ref_date : datetime, optional
        Reference date to split past vs. future appointments. Used in simulation.
        Default: 2024-12-01

    seed : int
        Random seed for reproducibility.
        Default: 42

    noise : float
        Magnitude of random variation to introduce in population and attendance behavior.
        Default: 0.05

    median_lead_time : int
        Median number of days between scheduling and appointment date.
        Default: 10

    status_rates : dict, optional
        Dictionary of outcome probabilities:
        { "attended", "cancelled", "did not attend", "unknown" }
        If not provided, uses NHS defaults.

    rebook_category : str
        Rebooking intensity after cancellations. Options: {"min", "med", "max"}.
        Default: "med"

    check_in_time_mean : float
        Mean number of minutes before appointment that patients arrive.
        Default: -10

    visits_per_year : float
        Average number of visits per patient per year.
        Default: 1.2

    bin_size : int
        Size (in years) of age group intervals.
        Default: 5

    lower_cutoff : int
        Minimum age of patients to be included.
        Default: 15

    upper_cutoff : int
        Maximum age of patients to be included.
        Default: 90

    truncated : bool
        Whether to exclude patients below the lower_cutoff.
        Default: True

    first_attendance : float, optional
        Annual proportion of patients that are first-time visitors.
        If None, uses NHS default (0.325)

    age_gender_probs : pd.DataFrame, optional
        Demographic probabilities by age and sex. If None, uses NHS defaults.
    """

    def __init__(self,
                 date_ranges: Optional[List[Tuple[datetime, datetime]]] = None,
                 working_days: Optional[List[int]] = None,
                 working_hours: Optional[List[Tuple[int, int]]] = None,
                 appointments_per_hour: int = 4,
                 fill_rate: float = 0.9,
                 booking_horizon: int = 30,
                 ref_date: Optional[datetime] = None,
                 seed: int = 42,
                 noise: float = 0.05,
                 median_lead_time: int = 10,
                 status_rates: Optional[dict] = None,
                 rebook_category: str = "med",
                 check_in_time_mean: float = -10,
                 visits_per_year: float = 1.2,
                 bin_size: int = 5,
                 lower_cutoff: int = 15,
                 upper_cutoff: int = 90,
                 truncated: bool = True,
                 first_attendance: Optional[float] = None,
                 age_gender_probs: Optional[pd.DataFrame] = None):

        # Assign configuration parameters with fallbacks
        self.date_ranges = date_ranges or [(datetime(2015, 1, 1), datetime(2024, 12, 31))]
        self.working_days = working_days or [0, 1, 2, 3, 4]
        self.working_hours = working_hours or [(8, 18)]
        self.appointments_per_hour = appointments_per_hour
        self.fill_rate = fill_rate
        self.booking_horizon = booking_horizon
        self.ref_date = ref_date or datetime(2024, 12, 1)
        self.seed = seed
        self.noise = noise
        self.median_lead_time = median_lead_time

        self.status_rates = status_rates or DEFAULT_STATUS_RATES
        total = sum(self.status_rates.values())
        self.status_rates = {k: v / total for k, v in self.status_rates.items()}

        self.rebook_category = rebook_category
        if self.rebook_category == "min":
            self.rebook_ratio = 0
        elif self.rebook_category == "med":
            self.rebook_ratio = 0.5
        elif self.rebook_category == "max":
            self.rebook_ratio = 1
        else:
            raise ValueError("Invalid rebook_category: choose 'min', 'med', or 'max'.")

        self.check_in_time_mean = check_in_time_mean
        self.visits_per_year = visits_per_year
        self.bin_size = bin_size
        self.lower_cutoff = lower_cutoff
        self.upper_cutoff = upper_cutoff
        self.truncated = truncated
        self.first_attendance = first_attendance if first_attendance is not None else DEFAULT_FIRST_ATTENDANCE_RATIO
        self.age_gender_probs = pd.DataFrame(age_gender_probs or DEFAULT_AGE_GENDER_PROBS)

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

        # Ensure that 'ref_date + booking_horizon' is inside 'date_ranges'; if not, extend 'date_ranges'
        if self.ref_date + timedelta(days=self.booking_horizon) > max(end for _, end in self.date_ranges):
            self.date_ranges[-1] = (self.date_ranges[-1][0], self.ref_date + timedelta(days=self.booking_horizon))

        self.total_days = sum((end - start).days + 1 for start, end in self.date_ranges)
        self.slots_per_day = len(self.working_hours) * self.appointments_per_hour * (self.working_hours[0][1] - self.working_hours[0][0])
        self.working_days_count = sum(1 for date_day in pd.date_range(self.date_ranges[0][0], self.date_ranges[-1][1]) if date_day.weekday() in self.working_days)
        self.slots_per_week = self.slots_per_day * len(self.working_days)

        self.patients_df = pd.DataFrame()
        self.patient_id_counter = 1



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
        slots = []
        slot_id_counter = 1
        max_digits = len(str(self.total_days * self.slots_per_day)) + 1

        for start_date, end_date in self.date_ranges:
            for date_day in pd.date_range(start=start_date, end=end_date):
                if date_day.weekday() in self.working_days:
                    for start_hour, end_hour in self.working_hours:
                        for hour in range(start_hour, end_hour):
                            for m in range(0, 60, 60 // self.appointments_per_hour):
                                time_slot = datetime.combine(date_day, time(hour=hour, minute=m))
                                slots.append({
                                    "slot_id": str(slot_id_counter).zfill(max_digits),
                                    "appointment_date": date_day.normalize(),
                                    "appointment_time": time_slot.time(),
                                    "is_available": True
                                })
                                slot_id_counter += 1

        slots_df = pd.DataFrame(slots)
        slots_df["appointment_date"] = pd.to_datetime(slots_df["appointment_date"])
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
        - The average arrival time is 10 minutes before the scheduled slot.
        - A normal distribution controls variability in arrival offsets.

        Parameters
        ----------
        appointment_date : datetime.date
            Date of the scheduled appointment.
        appointment_time : datetime.time
            Scheduled start time of the appointment.

        Returns
        -------
        datetime
            Simulated check-in timestamp (may be before or after the scheduled time).
        """
        scheduled_dt = datetime.combine(appointment_date, appointment_time)

        # Empirical punctuality distribution
        mean_offset_min = self.check_in_time_mean  # e.g., -10 mins (early)
        std_offset_min = 9.8  # Derived from clinic-level std deviation

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
