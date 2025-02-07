package filelog

import (
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"os"
)

var (
	FileLogger zerolog.Logger
	logPath    = "trace.txt"
)

func SetupFileLogger() {
	f, err := os.Create(logPath)
	if err != nil {
		log.Fatal().Err(err).Msg("fail to create log file")
	}
	FileLogger = zerolog.New(f).With().Timestamp().Logger()
	FileLogger.Info().Str("path", logPath).Msg("file log configured")
}
