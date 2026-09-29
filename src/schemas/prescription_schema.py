from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class Medicine(BaseModel):
    medicine_name: Optional[str] = None
    quantity: Optional[str] = None
    strength: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    instruction: Optional[str] = None

    @field_validator(
        "medicine_name",
        "quantity",
        "strength",
        "frequency",
        "duration",
        "instruction",
        mode="before",
    )
    @classmethod
    def convert_to_string(cls, value):
        if value is None:
            return None

        return str(value)


class Prescription(BaseModel):
    patient_name: Optional[str] = None
    patient_age: Optional[str] = None
    patient_gender: Optional[str] = None
    prescription_date: Optional[str] = None

    doctor_name: Optional[str] = None
    doctor_registration_number: Optional[str] = None
    clinic_name: Optional[str] = None
    clinic_address: Optional[str] = None
    clinic_phone: Optional[str] = None

    medicines: List[Medicine] = Field(
        default_factory=list
    )

    global_instruction: Optional[str] = None

    @field_validator(
        "patient_name",
        "patient_age",
        "patient_gender",
        "prescription_date",
        "doctor_name",
        "doctor_registration_number",
        "clinic_name",
        "clinic_address",
        "clinic_phone",
        "global_instruction",
        mode="before",
    )
    @classmethod
    def convert_to_string(cls, value):
        if value is None:
            return None

        return str(value)