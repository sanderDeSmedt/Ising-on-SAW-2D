FROM --platform=${TARGETPLATFORM:-linux/amd64} gcr.io/distroless/base

ENV TZ Europe/Brussels

COPY build/service .
      
ENTRYPOINT ["./service"]