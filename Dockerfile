# 
FROM python:3.9

# 
WORKDIR /app

#
#COPY requirements.txt ./ 
COPY eva-api/requirements.txt ./

# 
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 
# COPY ./ ./
COPY eva-api/. ./

# 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
