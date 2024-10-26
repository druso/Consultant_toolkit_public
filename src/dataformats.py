
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class BatchRequestPayload:
    batch_id: str
    user_id: str
    session_id: str
    function: str
    batch_size: int
    input_file: str
    query_column: str
    response_column: str
    kwargs: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchRequestPayload':
        try:
            return cls(
                batch_id=data['batch_id'],
                user_id=data['user_id'],
                session_id=data['session_id'],
                function=data['function'],
                batch_size=data['batch_size'],
                input_file=data['input_file'],
                query_column=data['query_column'],
                response_column=data['response_column'],
                kwargs=data['kwargs']
            )
        except KeyError as e:
            raise ValueError(f"Invalid payload structure: missing {e}") from e

@dataclass
class BatchSummaryPayload:
    batch_id: str
    user_id: str
    session_id: str
    function: str
    batch_size: 0
    input_file: str
    query_column: str
    response_column: str
    filename: str = ""
    schedule_time: str = ""    
    start_time: str = ""
    end_time: str = ""
    status: str = "PENDING"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_request_payload(cls, payload: BatchRequestPayload, status: str = "PENDING"):
        return cls(
            batch_id=payload.batch_id,
            user_id=payload.user_id,
            session_id=payload.session_id,
            function=payload.function,
            batch_size=payload.batch_size,
            input_file=payload.input_file,
            query_column=payload.query_column,
            response_column=payload.response_column,
            status=status
        )
    #I should add a calculated column for the time it takes to process the batch
    #I should add a column for the position of the resulted file (now constructed in streamlit_interface)




